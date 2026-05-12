# INSERT_YOUR_CODE
import os
import shutil
from pathlib import Path

# Define relevant directories
base_dir = Path("/root/data/shadowmagic_experiment/flymyai-lora-trainer")
screen_control_images_dir = base_dir / "screen_control_images"
screen_images_dir = base_dir / "screen_images"
src_dirs = {
    'dungeon': Path("/root/data/screen_dungeon"),
    'JumGweGong': Path("/root/data/screen_JumGweGong")
}

# Create target directories if they don't exist
screen_control_images_dir.mkdir(parents=True, exist_ok=True)
screen_images_dir.mkdir(parents=True, exist_ok=True)

for domain, src_dir in src_dirs.items():
    for root, _, files in os.walk(src_dir):
        for fname in files:
            # Process _line.png files -> control images
            if fname.endswith("_line.png"):
                prefix = f"{domain}_"
                src_path = Path(root) / fname
                dst_filename = prefix + fname
                dst_path = screen_control_images_dir / dst_filename
                shutil.copy2(src_path, dst_path)
            # Process screen_processed.png files -> images
            elif fname.endswith("screen_processed.png"):
                prefix = f"{domain}_"
                src_path = Path(root) / fname
                # Replace ending with line.png
                if fname.endswith("screen_processed.png"):
                    base = fname[:-len("screen_processed.png")]
                    new_fname = base + "line.png"
                    dst_filename = prefix + new_fname
                    dst_path = screen_images_dir / dst_filename
                    shutil.copy2(src_path, dst_path)
