#!/usr/bin/env python3
import os
import shutil
from pathlib import Path
import fnmatch

SRC_DIR = Path('/root/data/shadowmagic')
IMAGES_DIR = Path('/root/data/shadowmagic_experiment/images')
CONTROL_DIR = Path('/root/data/shadowmagic_experiment/control_images')

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CONTROL_DIR.mkdir(parents=True, exist_ok=True)

# Step 1: find png files under SRC_DIR whose filename includes 'shadow' and one of back,left,right,top
keywords = ['back', 'left', 'right', 'top']
matched_files = []
for root, dirs, files in os.walk(SRC_DIR):
    for f in files:
        if not f.lower().endswith('.png'):
            continue
        name = f.lower()
        if 'shadow' in name and any(k in name for k in keywords):
            matched_files.append(Path(root) / f)

copied_to_images = []
for p in matched_files:
    dest = IMAGES_DIR / p.name
    try:
        shutil.copy2(p, dest)
        copied_to_images.append(dest)
    except Exception as e:
        print(f"Failed to copy {p} -> {dest}: {e}")

print(f"Step1: Found {len(matched_files)} matching source pngs, copied {len(copied_to_images)} to {IMAGES_DIR}")

# Step 2: For each file in images dir, find unique file in SRC_DIR whose filename contains the file's first 4 chars and 'line'
images_now = sorted([p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower()=='.png'])

report = []
for img_path in images_now:
    A = img_path.name
    prefix = A[:4]
    # build pattern: *{prefix}*line*.png (case-insensitive)
    matches = []
    for root, dirs, files in os.walk(SRC_DIR):
        for f in files:
            if not f.lower().endswith('.png'):
                continue
            fname_lower = f.lower()
            if prefix.lower() in fname_lower and 'line' in fname_lower:
                matches.append(Path(root) / f)
    if len(matches) == 1:
        src = matches[0]
        dest = CONTROL_DIR / A  # rename to A
        try:
            shutil.copy2(src, dest)
            report.append((A, 'ok', str(src), str(dest)))
        except Exception as e:
            report.append((A, 'copy-failed', str(src), str(e)))
    else:
        if len(matches) == 0:
            report.append((A, 'no-match', None, None))
        else:
            # multiple matches
            report.append((A, 'multiple', [str(m) for m in matches], None))

ok = sum(1 for r in report if r[1] == 'ok')
no = sum(1 for r in report if r[1] == 'no-match')
mul = sum(1 for r in report if r[1] == 'multiple')
fail = sum(1 for r in report if r[1] == 'copy-failed')

print(f"Step2: Processed {len(images_now)} images in {IMAGES_DIR}: ok={ok}, no-match={no}, multiple={mul}, copy-failed={fail}")

# Print details for problems
if no > 0 or mul > 0 or fail > 0:
    print('\nDetails:')
    for r in report:
        if r[1] != 'ok':
            print(r)

# Summary list of copied control files (names)
copied_controls = sorted([p.name for p in CONTROL_DIR.iterdir() if p.is_file() and p.suffix.lower()=='.png'])
print(f"Copied control files into {CONTROL_DIR}: {len(copied_controls)} files")
for n in copied_controls[:200]:
    print(n)
