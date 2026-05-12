import argparse
import copy
import logging
import math
import os
import shutil

import bitsandbytes as bnb
import datasets
import diffusers
import numpy as np
import torch
import transformers
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from image_datasets.control_dataset import combined_loader, loader, screen_layer_loader
from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from quanto import freeze, qfloat8, quantize
from tqdm.auto import tqdm

logger = get_logger(__name__, log_level="INFO")

import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TRAINING_LOG_ROWS = []
VALIDATION_LOG_ROWS = []


def _trim_log_buffer(buffer, max_size):
    if max_size is None or max_size <= 0:
        return
    extra = len(buffer) - max_size
    if extra > 0:
        del buffer[:extra]


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    return parser.parse_args()

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height, None


def prompt_from_filename(filename: str) -> str:
    stem = os.path.splitext(filename)[0].lower()
    keyword_mapping = {
        "back": "BACK shadow",
        "left": "LEFT shadow",
        "right": "RIGHT shadow",
        "top": "TOP shadow",
        "front": "FRONT shadow",
        "bottom": "BOTTOM shadow",
        "side": "SIDE shadow",
        "corner": "CORNER shadow",
        "dungeon": "FIRST_STYLE screen layer",
        "JumGweGong": "SECOND_STYLE screen layer",
    }
    for keyword, prompt in keyword_mapping.items():
        if keyword in stem:
            return prompt
    return "screen layer"


def read_prompt_text(image_name: str, img_dir: str) -> str:
    txt_path = os.path.join(img_dir, os.path.splitext(image_name)[0] + ".txt")
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as fp:
            prompt = fp.read().strip()
            if prompt:
                return prompt
    return prompt_from_filename(image_name)

def resolve_path(base_dir: str, path: str) -> str:
    if path is None:
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def log_validation(accelerator, unwrap_model, weight_dtype, args, cli_args, global_step, flux_transformer, vae):
    global VALIDATION_LOG_ROWS
    if not accelerator.is_main_process:
        return

    if args.report_to != "wandb" or not hasattr(args, "validation_steps"):
        return

    if global_step <= 0 or global_step % args.validation_steps != 0:
        return

    logger.info("Running validation...")

    # Extract LoRA state without moving model off GPU
    unwrapped_flux_transformer = unwrap_model(flux_transformer)
    
    # Get PEFT state dict directly without moving to CPU
    # This is safe because we only extract LoRA parameters
    with torch.no_grad():
        peft_state_dict = get_peft_model_state_dict(unwrapped_flux_transformer)
        # Convert to CPU for pipeline loading
        peft_state_dict = {k: v.cpu() for k, v in peft_state_dict.items()}
    
    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(peft_state_dict)

    # Add prefix for proper loading
    lora_state_dict_with_prefix = {f"transformer.{k}": v for k, v in flux_transformer_lora_state_dict.items()}

    # Free up GPU memory before loading validation pipeline
    # Use CPU offloading to avoid OOM while keeping models on GPU
    logger.info("Loading validation pipeline with CPU offloading to avoid OOM...")
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load validation pipeline with memory optimization - use CPU offloading
    validation_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
    )

    validation_pipeline.load_lora_weights(lora_state_dict_with_prefix)
    
    # Use sequential CPU offloading for validation pipeline to reduce memory usage
    # This automatically moves components to GPU only when needed
    # This keeps the training models on GPU untouched
    try:
        validation_pipeline.enable_sequential_cpu_offload()
        logger.info("Enabled sequential CPU offload for validation pipeline")
    except Exception as e:
        logger.warning(f"Could not enable CPU offload: {e}, using regular device placement")
        # Fallback to regular device placement
        validation_pipeline.to(accelerator.device)
    
    # Additional memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

    if hasattr(args, "validation_prompts") and hasattr(args, "validation_control_images"):
        for prompt, control_image_path in zip(args.validation_prompts, args.validation_control_images):
            control_image_path = resolve_path(os.path.dirname(cli_args.config), control_image_path)
            control_image = Image.open(control_image_path)
            # Handle alpha channel properly in validation too
            if control_image.mode in ("RGBA", "LA") or (
                hasattr(control_image, "info") and control_image.info.get("transparency") is not None
            ):
                bg = Image.new("RGB", control_image.size, (255, 255, 255))
                bg.paste(control_image, mask=control_image.split()[-1])
                control_image = bg
            else:
                control_image = control_image.convert("RGB")

            control_image.save("/data/debug_control_image_validation.png")
            print(f"\n[DEBUG] Validation logging to wandb: step={global_step}")
            print(f"  prompt='{prompt}'")
            print(f"  control_path={control_image_path}")
            arr = np.array(control_image)
            print(
                f"  control_image: shape={arr.shape} "
                f"dtype={arr.dtype} range=[{arr.min()}, {arr.max()}] "
                f"mean={arr.mean():.1f}"
            )

            with torch.no_grad():
                image = validation_pipeline(
                    prompt=prompt,
                    image=control_image,
                    num_inference_steps=getattr(args, "validation_inference_steps", 20),  # Reduced from 30 for memory
                    generator=torch.Generator(device=accelerator.device).manual_seed(
                        getattr(args, "seed", 42)
                    ),
                ).images[0]

            VALIDATION_LOG_ROWS.append(
                (global_step, control_image.copy(), prompt, image.copy())
            )
            _trim_log_buffer(VALIDATION_LOG_ROWS, getattr(args, "max_validation_log_rows", 8))

    if VALIDATION_LOG_ROWS:
        validation_table = wandb.Table(columns=["Step", "Control Image", "Prompt", "Generated Image"])
        for step_value, ctrl_img, prompt_text, gen_img in VALIDATION_LOG_ROWS:
            validation_table.add_data(step_value, wandb.Image(ctrl_img), prompt_text, wandb.Image(gen_img))
        try:
            tracker = accelerator.get_tracker("wandb")
            if tracker:
                tracker.log({"Validation Samples": validation_table})
        except Exception as e:
            logger.warning(f"Failed to log validation table to W&B: {e}")

    # Clean up validation pipeline - no need to restore models since they stayed on GPU
    del validation_pipeline, peft_state_dict, flux_transformer_lora_state_dict, lora_state_dict_with_prefix
    gc.collect()
    torch.cuda.empty_cache()


def main():
    # NCCL 설정 추가
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_SOCKET_IFNAME"] = "eth0"
    os.environ["NCCL_IB_TIMEOUT"] = "120"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"
    
    cli_args = parse_args()
    config_path = os.path.abspath(cli_args.config)
    args = OmegaConf.load(config_path)
    config_dir = os.path.dirname(config_path)

    args.output_dir = resolve_path(config_dir, args.output_dir)
    data_config = None
    if "data_config" in args:
        if "img_dir" in args.data_config:
            args.data_config.img_dir = resolve_path(config_dir, args.data_config.img_dir)
        if "control_dir" in args.data_config and args.data_config.control_dir is not None:
            args.data_config.control_dir = resolve_path(config_dir, args.data_config.control_dir)
        if "domains" in args.data_config and args.data_config.domains is not None:
            for domain in args.data_config.domains:
                if "img_dir" in domain:
                    domain.img_dir = resolve_path(config_dir, domain.img_dir)
                if "control_dir" in domain and domain.control_dir is not None:
                    domain.control_dir = resolve_path(config_dir, domain.control_dir)
        data_config = OmegaConf.to_container(args.data_config, resolve=True)
    else:
        raise ValueError("Configuration must include a `data_config` section.")

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    
    precompute_text = bool(getattr(args, "precompute_text_embeddings", False))
    precompute_image = bool(getattr(args, "precompute_image_embeddings", False))
    save_cache_on_disk = bool(getattr(args, "save_cache_on_disk", False))

    cache_dir = None
    if (precompute_text or precompute_image) and args.output_dir:
        cache_dir = os.path.join(args.output_dir, "cache")
        if accelerator.is_main_process:
            os.makedirs(cache_dir, exist_ok=True)
        accelerator.wait_for_everyone()

    domains_cfg = getattr(args.data_config, "domains", None)
    if domains_cfg is not None:
        # Multi-domain mode: image_filenames is only consumed by the precompute
        # branches below, which we don't support for combined training yet.
        if precompute_text or precompute_image:
            raise NotImplementedError(
                "precompute_text/image_embeddings is not supported when "
                "data_config.domains is set."
            )
        image_filenames = []
    else:
        image_filenames = sorted(
            name
            for name in os.listdir(args.data_config.img_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )
        if not image_filenames:
            raise ValueError(f"No training images found in {args.data_config.img_dir}")

    text_encoding_pipeline = QwenImageEditPlusPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )
    text_encoding_pipeline.to(accelerator.device)

    cached_text_embeddings = None
    text_cache_dir = None

    if precompute_text:
        if cache_dir is None:
            cache_dir = os.path.join(args.output_dir, "cache")
            if accelerator.is_main_process:
                os.makedirs(cache_dir, exist_ok=True)
            accelerator.wait_for_everyone()
        if save_cache_on_disk:
            text_cache_dir = os.path.join(cache_dir, "text_embs")
            if accelerator.is_main_process:
                os.makedirs(text_cache_dir, exist_ok=True)
        else:
            cached_text_embeddings = {}
        accelerator.wait_for_everyone()

        control_dir = getattr(args.data_config, "control_dir", None)
        with torch.no_grad():
            for image_name in tqdm(
                image_filenames,
                desc="Caching text embeddings",
                disable=not accelerator.is_local_main_process,
            ):
                control_path = (
                    os.path.join(control_dir, image_name)
                    if control_dir and os.path.exists(os.path.join(control_dir, image_name))
                    else os.path.join(args.data_config.img_dir, image_name)
                )
                if not os.path.exists(control_path):
                    logger.warning(f"Skipping text embedding for {image_name}: file not found.")
                    continue
                
                image = Image.open(control_path).convert("RGB")
                calculated_width, calculated_height, _ = calculate_dimensions(
                    1024 * 1024, image.size[0] / image.size[1]
                )
                prompt_image = text_encoding_pipeline.image_processor.resize(image, calculated_height, calculated_width)

                prompt_text = read_prompt_text(image_name, args.data_config.img_dir)
                prompt_text = prompt_text if prompt_text.strip() else " "

                prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                    image=prompt_image,
                    prompt=[prompt_text],
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )
                empty_embeds, empty_mask = text_encoding_pipeline.encode_prompt(
                    image=prompt_image,
                    prompt=[" "],
                    device=text_encoding_pipeline.device,
                    num_images_per_prompt=1,
                    max_sequence_length=1024,
                )

                prompt_embeds = prompt_embeds[0].to("cpu")
                prompt_embeds_mask = prompt_embeds_mask[0].to("cpu")
                empty_embeds = empty_embeds[0].to("cpu")
                empty_mask = empty_mask[0].to("cpu")

                base_key = os.path.splitext(image_name)[0]
                if save_cache_on_disk:
                    torch.save(
                        {"prompt_embeds": prompt_embeds, "prompt_embeds_mask": prompt_embeds_mask},
                        os.path.join(text_cache_dir, base_key + ".pt"),
                    )
                    torch.save(
                        {"prompt_embeds": empty_embeds, "prompt_embeds_mask": empty_mask},
                        os.path.join(text_cache_dir, base_key + "_empty.pt"),
                    )
                else:
                    cached_text_embeddings[f"{base_key}.txt"] = {
                        "prompt_embeds": prompt_embeds,
                        "prompt_embeds_mask": prompt_embeds_mask,
                    }
                    cached_text_embeddings[f"{base_key}.txtempty_embedding"] = {
                        "prompt_embeds": empty_embeds,
                        "prompt_embeds_mask": empty_mask,
                    }
        accelerator.wait_for_everyone()

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
    )
    vae.to(accelerator.device, dtype=weight_dtype)

    cached_image_embeddings = None
    cached_image_embeddings_control = None
    image_cache_dir = None
    control_cache_dir = None

    if precompute_image:
        if cache_dir is None:
            cache_dir = os.path.join(args.output_dir, "cache")
            if accelerator.is_main_process:
                os.makedirs(cache_dir, exist_ok=True)
            accelerator.wait_for_everyone()
        if save_cache_on_disk:
            image_cache_dir = os.path.join(cache_dir, "img_embs")
            control_cache_dir = os.path.join(cache_dir, "img_embs_control")
            if accelerator.is_main_process:
                os.makedirs(image_cache_dir, exist_ok=True)
                os.makedirs(control_cache_dir, exist_ok=True)
        else:
            cached_image_embeddings = {}
            cached_image_embeddings_control = {}
        accelerator.wait_for_everyone()

        with torch.no_grad():
            for image_name in tqdm(
                image_filenames,
                desc="Caching image latents",
                disable=not accelerator.is_local_main_process,
            ):
                img = Image.open(os.path.join(args.data_config.img_dir, image_name)).convert("RGB")
                calculated_width, calculated_height, _ = calculate_dimensions(
                    1024 * 1024, img.size[0] / img.size[1]
                )
                resized = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)
                arr = (np.array(resized).astype(np.float32) / 127.5) - 1.0
                tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
                pixel_values = tensor.unsqueeze(2).to(dtype=weight_dtype, device=accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.sample().to("cpu")[0]
                if save_cache_on_disk:
                    torch.save(latents, os.path.join(image_cache_dir, image_name + ".pt"))
                else:
                    cached_image_embeddings[image_name] = latents

            control_dir = getattr(args.data_config, "control_dir", None)
            for image_name in tqdm(
                image_filenames,
                desc="Caching control latents",
                disable=not accelerator.is_local_main_process,
            ):
                control_path = (
                    os.path.join(control_dir, image_name)
                    if control_dir and os.path.exists(os.path.join(control_dir, image_name))
                    else os.path.join(args.data_config.img_dir, image_name)
                )
                if not os.path.exists(control_path):
                    logger.warning(f"Skipping control latent for {image_name}: file not found.")
                    continue

                img = Image.open(control_path).convert("RGB")
                calculated_width, calculated_height, _ = calculate_dimensions(
                    1024 * 1024, img.size[0] / img.size[1]
                )
                resized = text_encoding_pipeline.image_processor.resize(img, calculated_height, calculated_width)
                arr = (np.array(resized).astype(np.float32) / 127.5) - 1.0
                tensor = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
                pixel_values = tensor.unsqueeze(2).to(dtype=weight_dtype, device=accelerator.device)
                latents = vae.encode(pixel_values).latent_dist.sample().to("cpu")[0]
                if save_cache_on_disk:
                    torch.save(latents, os.path.join(control_cache_dir, image_name + ".pt"))
                else:
                    cached_image_embeddings_control[image_name] = latents

        vae.to("cpu")
        torch.cuda.empty_cache()

    needs_text_encoder_during_training = not precompute_text
    if not needs_text_encoder_during_training:
        text_encoding_pipeline.to("cpu")
        torch.cuda.empty_cache()
    
    if not needs_text_encoder_during_training:
        text_encoding_pipeline = None
        gc.collect()

    # Load transformer for LoRA training
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
    )
    if args.quantize:
        torch_dtype = weight_dtype
        device = accelerator.device
        all_blocks = list(flux_transformer.transformer_blocks)
        for block in tqdm(all_blocks):
            block.to(device, dtype=torch_dtype)
            quantize(block, weights=qfloat8)
            freeze(block)
            block.to('cpu')
        flux_transformer.to(device, dtype=torch_dtype)
        quantize(flux_transformer, weights=qfloat8)
        freeze(flux_transformer)
        logger.info("Applied 8-bit quantization to transformer blocks.")

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    if args.quantize:
        flux_transformer.to(accelerator.device)
    else:
        flux_transformer.to(accelerator.device, dtype=weight_dtype)
    
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    flux_transformer.add_adapter(lora_config)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        
        # 벡터화된 연산으로 변경
        indices = torch.cat([(schedule_timesteps == t).nonzero() for t in timesteps])
        sigma = sigmas[indices].flatten()
        
        # 차원 확장을 한번에 처리
        sigma = sigma.view(-1, *([1] * (n_dim - 1)))
        return sigma
    # 모델 그래디언트 설정 최적화
    flux_transformer.requires_grad_(False)
    
    # diffusers 모델용 그래디언트 체크포인팅 설정
    if hasattr(flux_transformer, "enable_gradient_checkpointing"):
        flux_transformer.enable_gradient_checkpointing()
    else:
        # 일반적인 gradient checkpointing fallback
        flux_transformer._set_gradient_checkpointing(True)
    
    # LoRA 파라미터만 학습하도록 설정
    for name, param in flux_transformer.named_parameters():
        param.requires_grad = "lora" in name
        if param.requires_grad:
            logger.debug(f"LoRA parameter enabled for training: {name}")
    
    # 학습 가능한 파라미터 수집
    trainable_params = [p for p in flux_transformer.parameters() if p.requires_grad]
    
    # 메모리 최적화를 위한 추가 설정
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # GPU 메모리 캐시 초기화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    logger.info(
        "Trainable parameters (LoRA): %.2f M",
        sum(p.numel() for p in trainable_params) / 1_000_000,
    )

    # DataLoader 설정 최적화
    dataset_kwargs = dict(data_config or {})

    # 안전한 기본값 설정
    if 'num_workers' not in dataset_kwargs:
        dataset_kwargs['num_workers'] = 0

    # Remove any DataLoader-only kwargs from dataset kwargs so they are not
    # accidentally forwarded into CustomScreenImageDataset.__init__.
    dl_num_workers = dataset_kwargs.pop('num_workers', 0)
    dataset_kwargs.pop('pin_memory', None)
    dataset_kwargs.pop('persistent_workers', None)

    # The loader() helper expects train_batch_size (from dataset_kwargs) and
    # num_workers as separate arguments. Pass remaining dataset kwargs through.
    train_batch_size = dataset_kwargs.pop('train_batch_size', None)
    if train_batch_size is None:
        raise ValueError('train_batch_size must be set in data_config')

    domains = dataset_kwargs.pop("domains", None)
    if domains is not None:
        # Combined multi-domain training (shadow + screen).
        # img_dir/control_dir are owned per-domain; drop any top-level values so
        # they don't double-feed into the dataset constructors.
        dataset_kwargs.pop("img_dir", None)
        dataset_kwargs.pop("control_dir", None)
        train_dataloader = combined_loader(
            train_batch_size=train_batch_size,
            num_workers=dl_num_workers,
            domains=domains,
            cached_text_embeddings=cached_text_embeddings,
            cached_image_embeddings=cached_image_embeddings,
            cached_image_embeddings_control=cached_image_embeddings_control,
            text_cache_dir=text_cache_dir,
            image_cache_dir=image_cache_dir,
            control_cache_dir=control_cache_dir,
            **dataset_kwargs,
        )
    else:
        train_dataloader = screen_layer_loader(
            train_batch_size=train_batch_size,
            num_workers=dl_num_workers,
            cached_text_embeddings=cached_text_embeddings,
            cached_image_embeddings=cached_image_embeddings,
            cached_image_embeddings_control=cached_image_embeddings_control,
            text_cache_dir=text_cache_dir,
            image_cache_dir=image_cache_dir,
            control_cache_dir=control_cache_dir,
            **dataset_kwargs,
        )

    optimizer_cls = torch.optim.AdamW
    if args.adam8bit:
        optimizer = bnb.optim.Adam8bit(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
        )
    else:
        optimizer = optimizer_cls(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    initial_global_step = 0

    if args.report_to == "wandb":
        if accelerator.is_main_process:
            accelerator.init_trackers(args.tracker_project_name)

    # `train_batch_size` was popped from dataset_kwargs earlier and stored in
    # the local variable `train_batch_size` so use that value here.
    per_device_batch_size = train_batch_size
    args.train_batch_size = per_device_batch_size
    total_batch_size = per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {per_device_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor = 2 ** len(vae.temperal_downsample)
    base_latents_mean = torch.tensor(vae.config.latents_mean).view(1, 1, vae.config.z_dim, 1, 1)
    base_latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, 1, vae.config.z_dim, 1, 1)
    train_loss = 0.0
    train_iter = iter(train_dataloader)
    while global_step < args.max_train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        with accelerator.accumulate(flux_transformer):
            prompts = batch["prompt"]
            if isinstance(prompts, tuple):
                prompts = list(prompts)

            prompt_embeds = batch.get("prompt_embeds")
            prompt_embeds_mask = batch.get("prompt_embeds_mask")
            prompt_images = None
            
            if needs_text_encoder_during_training:
                prompt_images = []
                batch_control_paths = batch.get("control_path")
                for i, image_name in enumerate(batch["image_name"]):
                    if batch_control_paths is not None:
                        control_path = batch_control_paths[i]
                    else:
                        control_path = (
                            os.path.join(args.data_config.control_dir, image_name)
                            if getattr(args.data_config, "control_dir", None)
                            and os.path.exists(os.path.join(args.data_config.control_dir, image_name))
                            else os.path.join(args.data_config.img_dir, image_name)
                        )
                    if not os.path.exists(control_path):
                        raise FileNotFoundError(f"Control image for prompt not found: {control_path}")
                    with Image.open(control_path) as img:
                        # FIX: 프롬프트 인코딩용 이미지는 작게 리사이즈 (토큰 수 감소)
                        # 원본 크기 이미지 → 매우 많은 토큰 (screen_images: 3313+)
                        # 512x512로 리사이즈 → 합리적인 토큰 수 (images와 유사)
                        rgb_img = img.convert("RGB")
                        max_size = 512
                        if max(rgb_img.size) > max_size:
                            ratio = max_size / max(rgb_img.size)
                            new_size = tuple(int(dim * ratio) for dim in rgb_img.size)
                            rgb_img = rgb_img.resize(new_size, Image.Resampling.LANCZOS)
                        prompt_images.append(rgb_img)

            if prompt_embeds is not None and prompt_embeds_mask is not None:
                prompt_embeds = prompt_embeds.to(device=accelerator.device, dtype=weight_dtype)
                prompt_embeds_mask = prompt_embeds_mask.to(device=accelerator.device)
            else:
                if not needs_text_encoder_during_training or text_encoding_pipeline is None:
                    raise RuntimeError("Text encoder is required for training but embeddings were not precomputed.")
                with torch.no_grad():
                    prompt_embeds, prompt_embeds_mask = text_encoding_pipeline.encode_prompt(
                        prompt=prompts,
                        image=prompt_images,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        max_sequence_length=1024,
                    )

            image_batch = batch["image"]
            control_batch = batch["control_image"]

            with torch.no_grad():
                # DEBUG: 이미지 크기 확인
                if global_step == 0 or global_step % 10 == 0:
                    logger.info(f"[DEBUG] Step {global_step}: image_batch.shape = {image_batch.shape}")
                    logger.info(f"[DEBUG] Step {global_step}: control_batch.shape = {control_batch.shape}")
                
                if image_batch.ndim == 5:
                    pixel_latents = image_batch.to(device=accelerator.device, dtype=weight_dtype)
                else:
                    pixel_values = image_batch.to(device=accelerator.device, dtype=weight_dtype).unsqueeze(2)
                    pixel_latents = vae.encode(pixel_values).latent_dist.sample()

                if control_batch.ndim == 5:
                    control_latents = control_batch.to(device=accelerator.device, dtype=weight_dtype)
                else:
                    control_values = control_batch.to(device=accelerator.device, dtype=weight_dtype).unsqueeze(2)
                    control_latents = vae.encode(control_values).latent_dist.sample()

                pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                control_latents = control_latents.permute(0, 2, 1, 3, 4)

                latents_mean = base_latents_mean.to(pixel_latents.device, pixel_latents.dtype)
                latents_std = base_latents_std.to(pixel_latents.device, pixel_latents.dtype)
                pixel_latents = (pixel_latents - latents_mean) * latents_std
                control_latents = (control_latents - latents_mean) * latents_std

                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

            sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
            packed_noisy_model_input = QwenImageEditPlusPipeline._pack_latents(
                noisy_model_input,
                bsz,
                noisy_model_input.shape[2],
                noisy_model_input.shape[3],
                noisy_model_input.shape[4],
            )
            packed_control_latents = QwenImageEditPlusPipeline._pack_latents(
                control_latents,
                bsz,
                control_latents.shape[2],
                control_latents.shape[3],
                control_latents.shape[4],
            )
            img_shapes = [
                [
                    (1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                    (1, control_latents.shape[3] // 2, control_latents.shape[4] // 2),
                ]
                for _ in range(bsz)
            ]
            packed_noisy_model_input_concated = torch.cat(
                [packed_noisy_model_input, packed_control_latents], dim=1
            )
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).long().tolist()
            
            # DEBUG: img_shapes와 txt_seq_lens 확인
            if global_step % 10 == 0:
                logger.info(f"[DEBUG] Step {global_step}: img_shapes = {img_shapes}")
                logger.info(f"[DEBUG] Step {global_step}: txt_seq_lens = {txt_seq_lens}")
                logger.info(f"[DEBUG] Step {global_step}: packed_noisy_model_input_concated.shape = {packed_noisy_model_input_concated.shape}")
                logger.info(f"[DEBUG] Step {global_step}: prompt_embeds.shape = {prompt_embeds.shape}")
            
            model_pred = flux_transformer(
                hidden_states=packed_noisy_model_input_concated,
                timestep=timesteps / 1000,
                guidance=None,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]
            model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

            model_pred = QwenImageEditPlusPipeline._unpack_latents(
                model_pred,
                height=noisy_model_input.shape[3] * vae_scale_factor,
                width=noisy_model_input.shape[4] * vae_scale_factor,
                vae_scale_factor=vae_scale_factor,
            )
            weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
            target = noise - pixel_latents
            target = target.permute(0, 2, 1, 3, 4)
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()
            per_device_batch = pixel_latents.shape[0]
            avg_loss = accelerator.gather(loss.repeat(per_device_batch)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1
            accelerator.log(
                {"train_loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]},
                step=global_step,
            )
            train_loss = 0.0

            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")

                try:
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)
                except Exception:
                    pass
                unwrapped_flux_transformer = unwrap_model(flux_transformer)
                flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_flux_transformer)
                )

                QwenImageEditPlusPipeline.save_lora_weights(
                    save_path,
                    flux_transformer_lora_state_dict,
                    safe_serialization=True,
                )

                logger.info(f"Saved state to {save_path}")

            image_logging_steps = getattr(args, "image_logging_steps", 50)
            if image_logging_steps > 0 and global_step > 0 and global_step % image_logging_steps == 0:
                # Synchronous logging: ensure all ranks wait, then main process does logging
                accelerator.wait_for_everyone()
                try:
                    if accelerator.is_main_process:
                        logger.info("Logging training images...")

                        # Extract LoRA state without moving model off GPU
                        unwrapped_flux_transformer = unwrap_model(flux_transformer)
                        
                        # Get PEFT state dict directly without moving to CPU
                        with torch.no_grad():
                            peft_state_dict = get_peft_model_state_dict(unwrapped_flux_transformer)
                            # Convert to CPU for pipeline loading
                            peft_state_dict = {k: v.cpu() for k, v in peft_state_dict.items()}
                        
                        flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(peft_state_dict)

                        # Add prefix for proper loading
                        lora_state_dict_with_prefix = {
                            f"transformer.{k}": v for k, v in flux_transformer_lora_state_dict.items()
                        }

                        # Free up GPU memory before loading log pipeline
                        # Use CPU offloading to avoid OOM while keeping training models on GPU
                        logger.info("Loading logging pipeline with CPU offloading to avoid OOM...")
                        gc.collect()
                        torch.cuda.empty_cache()

                        log_pipeline = QwenImageEditPlusPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            torch_dtype=weight_dtype,
                        )
                        log_pipeline.load_lora_weights(lora_state_dict_with_prefix)
                        
                        # Use sequential CPU offloading for log pipeline to reduce memory usage
                        # This automatically moves components to GPU only when needed
                        # This keeps the training models on GPU untouched
                        try:
                            log_pipeline.enable_sequential_cpu_offload()
                            logger.info("Enabled sequential CPU offload for logging pipeline")
                        except Exception as e:
                            logger.warning(f"Could not enable CPU offload: {e}, using regular device placement")
                            # Fallback to regular device placement
                            log_pipeline.to(accelerator.device)
                        
                        # Additional memory cleanup
                        gc.collect()
                        torch.cuda.empty_cache()

                        # Use data from the current batch
                        prompt = batch["prompt"][0]
                        image_name = batch["image_name"][0]

                        batch_control_paths = batch.get("control_path")
                        if batch_control_paths is not None:
                            control_path = batch_control_paths[0]
                        else:
                            control_dir = getattr(args.data_config, "control_dir", None)
                            control_path = (
                                os.path.join(control_dir, image_name)
                                if control_dir and os.path.exists(os.path.join(control_dir, image_name))
                                else os.path.join(args.data_config.img_dir, image_name)
                            )
                        control_image = Image.open(control_path)
                        # Handle alpha channel properly
                        if control_image.mode in ("RGBA", "LA") or (
                            hasattr(control_image, "info") and 
                            control_image.info.get("transparency") is not None
                        ):
                            bg = Image.new("RGB", control_image.size, (255, 255, 255))
                            bg.paste(control_image, mask=control_image.split()[-1])
                            control_image = bg
                        else:
                            control_image = control_image.convert("RGB")
                        
                        # Save debug image and print stats
                        control_image.save("/data/debug_control_image_training.png")
                        print(f"\n[DEBUG] Logging to wandb: step={global_step}")
                        print(f"  prompt='{prompt}'")
                        print(f"  control_path={control_path}")
                        arr = np.array(control_image)
                        print(
                            f"  control_image: shape={arr.shape} "
                            f"dtype={arr.dtype} range=[{arr.min()}, {arr.max()}] "
                            f"mean={arr.mean():.1f}"
                        )

                        with torch.no_grad():
                            image = log_pipeline(
                                prompt=prompt,
                                image=control_image,
                                num_inference_steps=getattr(args, "log_inference_steps", 20),  # Reduced from 30 for memory
                                generator=torch.Generator(device=accelerator.device).manual_seed(
                                    getattr(args, "seed", 42)
                                ),
                            ).images[0]

                        try:
                            tracker = accelerator.get_tracker("wandb")
                            if tracker:
                                global TRAINING_LOG_ROWS
                                TRAINING_LOG_ROWS.append(
                                    (global_step, control_image.copy(), prompt, image.copy())
                                )
                                _trim_log_buffer(TRAINING_LOG_ROWS, getattr(args, "max_training_log_rows", 8))
                                training_table = wandb.Table(
                                    columns=["Step", "Control Image", "Prompt", "Generated Image"]
                                )
                                for step_value, ctrl_img, prompt_text, gen_img in TRAINING_LOG_ROWS:
                                    training_table.add_data(
                                        step_value,
                                        wandb.Image(ctrl_img),
                                        prompt_text,
                                        wandb.Image(gen_img),
                                    )
                                tracker.log({"Training Samples": training_table})
                        except Exception as e:
                            logger.warning(f"Failed to log training image to W&B: {e}")

                        # Cleanup - no need to restore models since they stayed on GPU
                        del log_pipeline, peft_state_dict, flux_transformer_lora_state_dict, lora_state_dict_with_prefix
                        gc.collect()
                        torch.cuda.empty_cache()
                except Exception as log_exc:
                    logger.warning(f"Skipping training image logging due to error: {log_exc}")
                finally:
                    accelerator.wait_for_everyone()

            if hasattr(args, "validation_steps") and args.validation_steps > 0 and global_step % args.validation_steps == 0:
                # synchronize before/after validation to avoid DDP collectives ordering issues
                accelerator.wait_for_everyone()
                try:
                    log_validation(
                        accelerator,
                        unwrap_model,
                        weight_dtype,
                        args,
                        cli_args,
                        global_step,
                        flux_transformer,
                        vae,
                    )
                except Exception as val_exc:
                    logger.warning(f"Validation logging skipped due to error: {val_exc}")
                finally:
                    accelerator.wait_for_everyone()

        logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
