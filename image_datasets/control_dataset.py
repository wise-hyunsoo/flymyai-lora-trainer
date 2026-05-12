import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

def _invert_and_normalize_image(array: np.ndarray) -> np.ndarray:
    array = 255 - array
    return (array.astype(np.float32) / 127.5) - 1.0

def _normalize_image(array: np.ndarray) -> np.ndarray:
    # 반전 없이 [-1, 1] 정규화만
    return (array.astype(np.float32) / 127.5) - 1.0

def _image_to_tensor(image: Image.Image, invert: bool = False) -> torch.Tensor:
    array = np.array(image)
    if array.ndim == 2:  # grayscale → RGB 3채널
        array = np.stack([array, array, array], axis=-1)

    if invert:
        array = _invert_and_normalize_image(array)
    else:
        array = _normalize_image(array)

    tensor = torch.from_numpy(array).permute(2, 0, 1)  # (C,H,W)
    return tensor

def throw_one(probability: float) -> int:
    return 1 if random.random() < probability else 0


def image_resize(img: Image.Image, max_size: int = 512, maintain_aspect_ratio: bool = True) -> Image.Image:
    """
    이미지를 리사이즈합니다.
    
    Args:
        img: 리사이즈할 이미지
        max_size: 목표 크기
        maintain_aspect_ratio: True면 aspect ratio 유지, False면 정사각형으로 강제
    
    Returns:
        리사이즈된 이미지
    """
    if maintain_aspect_ratio:
        # 기존 동작: aspect ratio 유지
        w, h = img.size
        if w >= h:
            new_w = max_size
            new_h = int((max_size / w) * h)
        else:
            new_h = max_size
            new_w = int((max_size / h) * w)
        return img.resize((new_w, new_h))
    else:
        # 새로운 동작: 정사각형으로 강제 (배치 크기 일관성 보장)
        return img.resize((max_size, max_size))


def c_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def crop_to_aspect_ratio(
    image: Image.Image, 
    ratio: str = "16:9", 
    random_crop: bool = True, 
    crop_box: Optional[tuple] = None
) -> tuple[Image.Image, tuple]:
    """
    이미지를 목표 aspect ratio로 crop합니다.
    
    Args:
        image: crop할 이미지
        ratio: 목표 aspect ratio ("16:9", "4:3", "1:1")
        random_crop: True면 랜덤 위치, False면 center crop
        crop_box: 이미 계산된 crop box를 제공하면 그것을 사용 (left, top, right, bottom)
    
    Returns:
        (cropped_image, crop_box): crop된 이미지와 사용된 crop box
    """
    width, height = image.size
    
    # crop_box가 주어지면 그것을 사용
    if crop_box is not None:
        return image.crop(crop_box), crop_box
    
    # crop_box를 계산
    ratio_map = {"16:9": (16, 9), "4:3": (4, 3), "1:1": (1, 1), "9:16": (9, 16), "3:4": (3, 4)}
    target_w, target_h = ratio_map[ratio]
    target_ratio_value = target_w / target_h

    current_ratio = width / height

    if current_ratio > target_ratio_value:
        # 좌우를 자름 (세로는 그대로 유지)
        new_width = int(height * target_ratio_value)
        max_offset = width - new_width
        if random_crop:
            offset = random.randint(0, max_offset) if max_offset > 0 else 0
        else:
            offset = max_offset // 2  # center crop
        crop_box = (offset, 0, offset + new_width, height)
    else:
        # 상하를 자름 (가로는 그대로 유지)
        new_height = int(width / target_ratio_value)
        max_offset = height - new_height
        if random_crop:
            offset = random.randint(0, max_offset) if max_offset > 0 else 0
        else:
            offset = max_offset // 2  # center crop
        crop_box = (0, offset, width, offset + new_height)

    return image.crop(crop_box), crop_box


def _ensure_multiple_of_32(image: Image.Image) -> Image.Image:
    width, height = image.size
    width = (width // 32) * 32
    height = (height // 32) * 32
    return image.resize((max(32, width), max(32, height)))


class CustomImageDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        img_size: int = 512,
        caption_type: str = "txt",
        random_ratio: bool = False,
        caption_dropout_rate: float = 0.1,
        cached_text_embeddings: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        cached_image_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        control_dir: Optional[str] = None,
        cached_image_embeddings_control: Optional[Dict[str, torch.Tensor]] = None,
        text_cache_dir: Optional[str] = None,
        image_cache_dir: Optional[str] = None,
        control_cache_dir: Optional[str] = None,
    ) -> None:
        self.image_paths = sorted(
            os.path.join(img_dir, name)
            for name in os.listdir(img_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )
        if not self.image_paths:
            raise ValueError(f"No image files found in directory: {img_dir}")

        self.caption_root = img_dir
        self.control_dir = control_dir
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate

        self.cached_text_embeddings = cached_text_embeddings
        self.text_cache_dir = text_cache_dir
        self.cached_image_embeddings = cached_image_embeddings
        self.image_cache_dir = image_cache_dir
        self.cached_control_image_embeddings = cached_image_embeddings_control
        self.control_cache_dir = control_cache_dir

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def _generate_prompt_from_filename(filename: str) -> str:
        filename_without_ext = os.path.splitext(filename)[0].lower()
        keyword_mapping = {
            "back": "BACK shadow",
            "left": "LEFT shadow",
            "right": "RIGHT shadow",
            "top": "TOP shadow",
            "front": "FRONT shadow",
            "bottom": "BOTTOM shadow",
            "side": "SIDE shadow",
            "corner": "CORNER shadow",
        }

        for keyword, prompt in keyword_mapping.items():
            if keyword in filename_without_ext:
                return prompt
        return "shadow"

    def _read_prompt(self, image_name: str) -> str:
        # HOTFIX: screen_images용 - txt 파일 완전히 무시, 파일명만 사용
        return self._generate_prompt_from_filename(image_name)
        # 원래 코드 (주석 처리):
        # if self.caption_type == "txt":
        #     caption_path = os.path.join(self.caption_root, os.path.splitext(image_name)[0] + ".txt")
        #     if os.path.exists(caption_path):
        #         with open(caption_path, encoding="utf-8") as fp:
        #             prompt = fp.read().strip()
        #             if prompt:
        #                 return prompt
        # return self._generate_prompt_from_filename(image_name)

    def _maybe_get_cached_text(self, image_name: str, use_empty_prompt: bool):
        base_key = os.path.splitext(image_name)[0]

        if self.cached_text_embeddings is not None:
            cache_key = f"{base_key}.txt" + ("empty_embedding" if use_empty_prompt else "")
            entry = self.cached_text_embeddings.get(cache_key)
            if entry is None:
                return None, None
            return entry["prompt_embeds"].clone(), entry["prompt_embeds_mask"].clone()

        if self.text_cache_dir is not None:
            filename = base_key + ("_empty.pt" if use_empty_prompt else ".pt")
            cache_path = os.path.join(self.text_cache_dir, filename)
            if os.path.exists(cache_path):
                entry = torch.load(cache_path, map_location="cpu")
                return entry["prompt_embeds"], entry["prompt_embeds_mask"]

        return None, None

    @staticmethod
    def _maybe_get_cached_latent(
        cache_dict: Optional[Dict[str, torch.Tensor]],
        cache_dir: Optional[str],
        image_name: str,
    ) -> Optional[torch.Tensor]:
        if cache_dict is not None:
            latent = cache_dict.get(image_name)
            if latent is None:
                return None
            return latent.clone()

        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, image_name + ".pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path, map_location="cpu")

        return None

    def _prepare_pil_image(
        self, 
        path: str, 
        ratio: str, 
        crop_box: Optional[tuple] = None
    ) -> tuple[Image.Image, Optional[tuple]]:
        """
        이미지를 준비합니다. crop_box가 주어지면 같은 위치에서 crop합니다.
        
        Returns:
            (prepared_image, crop_box): 준비된 이미지와 사용된 crop box
        """
        image = Image.open(path)
        if image.mode in ("RGBA", "LA") or (hasattr(image, "info") and image.info.get("transparency") is not None):
            bg = Image.new("RGB", image.size, (255, 255, 255))
            alpha = image.split()[-1]
            bg.paste(image, mask=alpha)
            image = bg
        else:
            image = image.convert("RGB")

        used_crop_box = None
        if ratio != "default":
            image, used_crop_box = crop_to_aspect_ratio(image, ratio, crop_box=crop_box)
        # aspect ratio를 유지하지 않고 정사각형으로 강제 리사이즈 (배치 일관성 보장)
        image = image_resize(image, self.img_size, maintain_aspect_ratio=False)
        image = _ensure_multiple_of_32(image)
        return image, used_crop_box

    @staticmethod
    def _apply_control_mask_to_image(base_image: Image.Image, control_image: Image.Image) -> Image.Image:
        # control 이미지를 [0,1] grayscale mask로 변환한 뒤 RGB base 이미지에 곱한다.
        base_array = np.asarray(base_image).astype(np.float32) / 255.0
        control_gray = control_image.convert("L")
        control_array = np.asarray(control_gray).astype(np.float32) / 255.0
        control_array = np.expand_dims(control_array, axis=-1)
        blended = np.clip(base_array * control_array, 0.0, 1.0)
        blended_uint8 = (blended * 255).astype(np.uint8)
        return Image.fromarray(blended_uint8, mode="RGB")

    def _load_and_process_image(self, path: str, ratio: str, invert: bool) -> torch.Tensor:
        image, _ = self._prepare_pil_image(path, ratio)
        return _image_to_tensor(image, invert=invert)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)

        ratio = "default"
        if self.random_ratio:
            ratio = random.choice(["16:9", "default", "1:1", "4:3"])

        control_source = (
            os.path.join(self.control_dir, image_name)
            if self.control_dir and os.path.exists(os.path.join(self.control_dir, image_name))
            else image_path
        )

        img_latents = self._maybe_get_cached_latent(self.cached_image_embeddings, self.image_cache_dir, image_name)
        control_latents = self._maybe_get_cached_latent(self.cached_control_image_embeddings, self.control_cache_dir, image_name)

        prepared_control_image: Optional[Image.Image] = None
        crop_box: Optional[tuple] = None

        if img_latents is None:
            # 첫 번째 이미지를 준비하면서 crop_box를 얻음
            prepared_image, crop_box = self._prepare_pil_image(image_path, ratio)
            # 두 번째 이미지는 같은 crop_box를 사용
            prepared_control_image, _ = self._prepare_pil_image(control_source, ratio, crop_box=crop_box)
            masked_image = self._apply_control_mask_to_image(prepared_image, prepared_control_image)
            img_latents = _image_to_tensor(masked_image, invert=False)

        if control_latents is None:
            if prepared_control_image is None:
                prepared_control_image, _ = self._prepare_pil_image(control_source, ratio, crop_box=crop_box)
            control_latents = _image_to_tensor(prepared_control_image, invert=False)

        prompt = self._read_prompt(image_name)
        use_empty_prompt = bool(throw_one(self.caption_dropout_rate))
        prompt_text = " " if use_empty_prompt else prompt

        prompt_embeds, prompt_embeds_mask = self._maybe_get_cached_text(image_name, use_empty_prompt)

        sample: Dict[str, torch.Tensor] = {
            "image": img_latents,
            "control_image": control_latents,
            "prompt": prompt_text,
            "image_name": image_name,
            "control_path": control_source,
            "img_dir": self.caption_root,
        }

        if prompt_embeds is not None and prompt_embeds_mask is not None:
            sample["prompt_embeds"] = prompt_embeds
            sample["prompt_embeds_mask"] = prompt_embeds_mask

        return sample


def loader(
    train_batch_size: int,
    num_workers: int,
    text_cache_dir: Optional[str] = None,
    image_cache_dir: Optional[str] = None,
    control_cache_dir: Optional[str] = None,
    **dataset_kwargs,
) -> DataLoader:
    dataset = CustomImageDataset(
        text_cache_dir=text_cache_dir,
        image_cache_dir=image_cache_dir,
        control_cache_dir=control_cache_dir,
        **dataset_kwargs,
    )
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)


def screen_layer_loader(
    train_batch_size: int,
    num_workers: int,
    text_cache_dir: Optional[str] = None,
    image_cache_dir: Optional[str] = None,
    control_cache_dir: Optional[str] = None,
    use_balanced_sampling: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    CustomScreenImageDataset을 위한 DataLoader를 생성합니다.
    
    Args:
        use_balanced_sampling: True면 클래스 균등 샘플링 사용 (WeightedRandomSampler)
    """
    dataset = CustomScreenImageDataset(
        text_cache_dir=text_cache_dir,
        image_cache_dir=image_cache_dir,
        control_cache_dir=control_cache_dir,
        **dataset_kwargs,
    )
    
    if use_balanced_sampling:
        # WeightedRandomSampler를 사용하여 균등 샘플링
        sample_weights = dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True  # 중복 허용 (오버샘플링)
        )
        return DataLoader(
            dataset, 
            batch_size=train_batch_size, 
            num_workers=num_workers, 
            sampler=sampler,  # sampler 사용 시 shuffle=False
            pin_memory=True
        )
    else:
        # 일반 랜덤 샘플링
        return DataLoader(
            dataset, 
            batch_size=train_batch_size, 
            num_workers=num_workers, 
            shuffle=True, 
            pin_memory=True
        )




class CustomScreenImageDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        img_size: int = 512,
        caption_type: str = "txt",
        random_ratio: bool = False,
        caption_dropout_rate: float = 0.1,
        cached_text_embeddings: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        cached_image_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        control_dir: Optional[str] = None,
        cached_image_embeddings_control: Optional[Dict[str, torch.Tensor]] = None,
        text_cache_dir: Optional[str] = None,
        image_cache_dir: Optional[str] = None,
        control_cache_dir: Optional[str] = None,
    ) -> None:
        self.image_paths = sorted(
            os.path.join(img_dir, name)
            for name in os.listdir(img_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )
        if not self.image_paths:
            raise ValueError(f"No image files found in directory: {img_dir}")

        self.caption_root = img_dir
        self.control_dir = control_dir
        self.img_size = img_size
        self.caption_type = caption_type
        self.random_ratio = random_ratio
        self.caption_dropout_rate = caption_dropout_rate

        self.cached_text_embeddings = cached_text_embeddings
        self.text_cache_dir = text_cache_dir
        self.cached_image_embeddings = cached_image_embeddings
        self.image_cache_dir = image_cache_dir
        self.cached_control_image_embeddings = cached_image_embeddings_control
        self.control_cache_dir = control_cache_dir
        
        # 균등 샘플링을 위한 클래스 레이블 및 가중치 계산
        self._compute_sample_weights()

    def __len__(self) -> int:
        return len(self.image_paths)
    
    def _compute_sample_weights(self) -> None:
        """각 샘플의 클래스 레이블을 식별하고, 균등 샘플링을 위한 가중치를 계산합니다."""
        # 각 이미지의 클래스 레이블 식별
        self.class_labels = []
        for path in self.image_paths:
            filename = os.path.basename(path).lower()
            if filename.startswith("dungeon"):
                self.class_labels.append(0)  # dungeon 클래스
            elif filename.startswith("jumgwegong"):
                self.class_labels.append(1)  # JumGweGong 클래스
            else:
                self.class_labels.append(2)  # 기타 클래스
        
        # 각 클래스의 샘플 개수 계산
        class_counts = {}
        for label in self.class_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # 각 클래스의 가중치 계산 (역수 사용)
        class_weights = {label: 1.0 / count for label, count in class_counts.items()}
        
        # 각 샘플의 가중치 할당
        self.sample_weights = [class_weights[label] for label in self.class_labels]
        
        print(f"[CustomScreenImageDataset] 클래스 분포: {class_counts}")
        print(f"[CustomScreenImageDataset] 클래스 가중치: {class_weights}")
    
    def get_sample_weights(self):
        """WeightedRandomSampler에서 사용할 샘플 가중치를 반환합니다."""
        return self.sample_weights

    @staticmethod
    def _generate_prompt_from_filename(filename: str) -> str:
        filename_without_ext = os.path.splitext(filename)[0].lower()
        keyword_mapping = {
            "dungeon": "ZIGZAG highlight brush",
            "jumgwegong": "CURVED highlight brush",
        }

        for keyword, prompt in keyword_mapping.items():
            if keyword in filename_without_ext:
                return prompt
        return "screen layer"

    def _read_prompt(self, image_name: str) -> str:
        # HOTFIX: screen_images용 - txt 파일 완전히 무시, 파일명만 사용
        return self._generate_prompt_from_filename(image_name)
        # 원래 코드 (주석 처리):
        # if self.caption_type == "txt":
        #     caption_path = os.path.join(self.caption_root, os.path.splitext(image_name)[0] + ".txt")
        #     if os.path.exists(caption_path):
        #         with open(caption_path, encoding="utf-8") as fp:
        #             prompt = fp.read().strip()
        #             if prompt:
        #                 return prompt
        # return self._generate_prompt_from_filename(image_name)

    def _maybe_get_cached_text(self, image_name: str, use_empty_prompt: bool):
        base_key = os.path.splitext(image_name)[0]

        if self.cached_text_embeddings is not None:
            cache_key = f"{base_key}.txt" + ("empty_embedding" if use_empty_prompt else "")
            entry = self.cached_text_embeddings.get(cache_key)
            if entry is None:
                return None, None
            return entry["prompt_embeds"].clone(), entry["prompt_embeds_mask"].clone()

        if self.text_cache_dir is not None:
            filename = base_key + ("_empty.pt" if use_empty_prompt else ".pt")
            cache_path = os.path.join(self.text_cache_dir, filename)
            if os.path.exists(cache_path):
                entry = torch.load(cache_path, map_location="cpu")
                return entry["prompt_embeds"], entry["prompt_embeds_mask"]

        return None, None

    @staticmethod
    def _maybe_get_cached_latent(
        cache_dict: Optional[Dict[str, torch.Tensor]],
        cache_dir: Optional[str],
        image_name: str,
    ) -> Optional[torch.Tensor]:
        if cache_dict is not None:
            latent = cache_dict.get(image_name)
            if latent is None:
                return None
            return latent.clone()

        if cache_dir is not None:
            cache_path = os.path.join(cache_dir, image_name + ".pt")
            if os.path.exists(cache_path):
                return torch.load(cache_path, map_location="cpu")

        return None

    def _prepare_pil_image(
        self, 
        path: str, 
        ratio: str, 
        crop_box: Optional[tuple] = None
    ) -> tuple[Image.Image, Optional[tuple]]:
        """
        이미지를 준비합니다. crop_box가 주어지면 같은 위치에서 crop합니다.
        
        Returns:
            (prepared_image, crop_box): 준비된 이미지와 사용된 crop box
        """
        image = Image.open(path)
        if image.mode in ("RGBA", "LA") or (hasattr(image, "info") and image.info.get("transparency") is not None):
            bg = Image.new("RGB", image.size, (255, 255, 255))
            alpha = image.split()[-1]
            bg.paste(image, mask=alpha)
            image = bg
        else:
            image = image.convert("RGB")

        used_crop_box = None
        if ratio != "default":
            image, used_crop_box = crop_to_aspect_ratio(image, ratio, crop_box=crop_box)
        # aspect ratio를 유지하지 않고 정사각형으로 강제 리사이즈 (배치 일관성 보장)
        image = image_resize(image, self.img_size, maintain_aspect_ratio=True)
        image = _ensure_multiple_of_32(image)
        return image, used_crop_box

    @staticmethod
    def _apply_control_mask_to_image(base_image: Image.Image, control_image: Image.Image) -> Image.Image:
        # inverted_control_image (검은색 배경에 흰색 선)를 받아서
        # 흰색 선분을 빨간색으로 변환한 후, base_image와 pixel-wise max 연산
        base_array = np.asarray(base_image).astype(np.uint8)
        
        # control_image를 grayscale로 변환하여 선분 영역 찾기
        control_gray = control_image.convert("L")
        control_array = np.asarray(control_gray).astype(np.uint8)
        
        # 흰색 선분을 빨간색으로 변환 (검은색 배경은 (0,0,0)으로 유지)
        red_line_array = np.zeros_like(base_array)
        red_line_array[:, :, 0] = control_array  # R 채널에 선분 강도
        red_line_array[:, :, 1] = 0              # G 채널은 0
        red_line_array[:, :, 2] = 0              # B 채널은 0
        
        # pixel-wise, channel-wise maximum
        result = np.maximum(base_array, red_line_array)
        
        return Image.fromarray(result, mode="RGB")

    def _load_and_process_image(self, path: str, ratio: str, invert: bool) -> torch.Tensor:
        image = self._prepare_pil_image(path, ratio)
        return _image_to_tensor(image, invert=invert)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)

        ratio = "default"
        if self.random_ratio:
            ratio = random.choice(["1:1", "3:4", "16:9", "9:16", "4:3", "default"])  # 다양한 aspect ratio로 크롭

        control_source = (
            os.path.join(self.control_dir, image_name)
            if self.control_dir and os.path.exists(os.path.join(self.control_dir, image_name))
            else image_path
        )

        img_latents = self._maybe_get_cached_latent(self.cached_image_embeddings, self.image_cache_dir, image_name)
        control_latents = self._maybe_get_cached_latent(self.cached_control_image_embeddings, self.control_cache_dir, image_name)

        prepared_control_image: Optional[Image.Image] = None
        crop_box: Optional[tuple] = None

        if img_latents is None:
            # 첫 번째 이미지를 준비하면서 crop_box를 얻음
            prepared_image, crop_box = self._prepare_pil_image(image_path, ratio)
            # 두 번째 이미지는 같은 crop_box를 사용
            prepared_control_image, _ = self._prepare_pil_image(control_source, ratio, crop_box=crop_box)
            # control image를 invert (흰색 바탕/검은 선 -> 검은 바탕/흰 선)
            # inverted_control_image = ImageOps.invert(prepared_control_image)
            masked_image = prepared_image # self._apply_control_mask_to_image(prepared_image, inverted_control_image)
            img_latents = _image_to_tensor(masked_image, invert=False)

        if control_latents is None:
            if prepared_control_image is None:
                prepared_control_image, _ = self._prepare_pil_image(control_source, ratio, crop_box=crop_box)
            control_latents = _image_to_tensor(prepared_control_image, invert=False)

        prompt = self._read_prompt(image_name)
        use_empty_prompt = bool(throw_one(self.caption_dropout_rate))
        prompt_text = " " if use_empty_prompt else prompt

        prompt_embeds, prompt_embeds_mask = self._maybe_get_cached_text(image_name, use_empty_prompt)

        sample: Dict[str, torch.Tensor] = {
            "image": img_latents,
            "control_image": control_latents,
            "prompt": prompt_text,
            "image_name": image_name,
            "control_path": control_source,
            "img_dir": self.caption_root,
        }

        if prompt_embeds is not None and prompt_embeds_mask is not None:
            sample["prompt_embeds"] = prompt_embeds
            sample["prompt_embeds_mask"] = prompt_embeds_mask

        return sample


def combined_loader(
    train_batch_size: int,
    num_workers: int,
    domains,
    text_cache_dir: Optional[str] = None,
    image_cache_dir: Optional[str] = None,
    control_cache_dir: Optional[str] = None,
    **shared_dataset_kwargs,
) -> DataLoader:
    """Build a DataLoader that mixes multiple domain datasets with per-domain
    equal sampling weight.

    `domains` is a list of dicts:
        [{"type": "shadow"|"screen", "img_dir": ..., "control_dir": ...}, ...]

    Each domain's sample weights are normalized so they sum to 1.0, making the
    expected number of samples per domain equal regardless of dataset size.
    For the screen domain, the per-class balanced weights from
    CustomScreenImageDataset are preserved within the domain.
    """
    sub_datasets = []
    sample_weights = []

    for domain in domains:
        domain = dict(domain)
        domain_type = domain.pop("type")
        domain_img_dir = domain.pop("img_dir")
        domain_control_dir = domain.pop("control_dir", None)

        if domain_type == "shadow":
            dataset_cls = CustomImageDataset
        elif domain_type == "screen":
            dataset_cls = CustomScreenImageDataset
        else:
            raise ValueError(f"Unknown domain type: {domain_type!r}")

        sub_dataset = dataset_cls(
            img_dir=domain_img_dir,
            control_dir=domain_control_dir,
            text_cache_dir=text_cache_dir,
            image_cache_dir=image_cache_dir,
            control_cache_dir=control_cache_dir,
            **shared_dataset_kwargs,
        )
        sub_datasets.append(sub_dataset)

        if hasattr(sub_dataset, "sample_weights"):
            sw = np.asarray(sub_dataset.sample_weights, dtype=np.float64)
        else:
            sw = np.ones(len(sub_dataset), dtype=np.float64)
        sw = sw / sw.sum()
        sample_weights.extend(sw.tolist())

    combined = ConcatDataset(sub_datasets)

    print(
        "[combined_loader] domain sizes:",
        [len(d) for d in sub_datasets],
        "total:",
        len(combined),
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(combined),
        replacement=True,
    )

    return DataLoader(
        combined,
        batch_size=train_batch_size,
        num_workers=num_workers,
        sampler=sampler,
        pin_memory=True,
    )
