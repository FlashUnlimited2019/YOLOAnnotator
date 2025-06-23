#!/usr/bin/env python3
import argparse
import os
import re
import random
import yaml
from tqdm import tqdm
from dds_cloudapi_sdk import Config, Client
from dds_cloudapi_sdk.tasks.v2_task import create_task_with_local_image_auto_resize
from PIL import Image, ImageDraw, ImageFont

# Define categories here
CATEGORIES = [
    'handbag', 'wallet',
    'jacket', 'dress', 'skirt', 'shoes', 'hat', 'shirt', 't-shirt',
]
CLASS_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
PIC_PATTERN = re.compile(r'^pic_(\d{6})\.(jpg|png)$', re.IGNORECASE)

# Category-specific confidence thresholds
CATEGORY_THRESHOLDS = {
    "handbag": 0.5,
    'jacket': 0.4,
}

# Synonym normalization mapping
# The key is the original label returned by detection (lowercase, hyphens removed, etc.),
# mapped to your desired unified class name.
LABEL_UNIFY = {
    't-shirt': 'shirt',
    'tshirt':  'shirt',
    'shirt':  'shirt',
}


def normalize_label(raw_label):
    """
    Normalize the raw label: lowercase, remove hyphens,
    and map to unified category if present in LABEL_UNIFY.
    Returns the unified label, or the normalized label if not mapped.
    """
    key = raw_label.strip().lower().replace('-', '')
    return LABEL_UNIFY.get(key, key)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatic annotation with GroundingDINO, YOLO format, and visualization with progress bar"
    )
    parser.add_argument('--input-dir', required=True, help='Directory with raw images')
    parser.add_argument('--output-dir', required=True, help='Root output dataset directory')
    parser.add_argument('--vis-dir', help='Directory to save visualized images', default=None)
    parser.add_argument('--token', required=True, help='DDS Cloud API token')
    parser.add_argument('--prompt', help='Text prompt for detection', default=None)
    parser.add_argument('--model', default='GroundingDino-1.6-Pro', help='Model name')
    parser.add_argument('--api-path', default='/v2/task/grounding_dino/detection', help='API path for detection')
    parser.add_argument('--bbox-threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.8, help='API NMS threshold (0 to disable)')
    return parser.parse_args()


def rename_and_copy(input_dir, images_out_dir):
    """
    1. In input_dir, rename files that do not match the pattern pic_xxxxxx.jpg/.png
       to the format pic_00000x in place.
    2. Copy all files (both already-correctly named and newly renamed) to images_out_dir.
    3. Copy files in the original order: correct ones first, then newly renamed ones appended.
    """
    import os, shutil

    os.makedirs(images_out_dir, exist_ok=True)
    # List all image files
    files = [f for f in os.listdir(input_dir)
             if f.lower().endswith(('.jpg', '.png'))]

    # Split into files that match the pattern and those that don't
    correct = []
    incorrect = []
    for f in files:
        if PIC_PATTERN.match(f):
            correct.append(f)
        else:
            incorrect.append(f)

    # Sort correctly named files by their numeric index
    correct.sort(key=lambda x: int(PIC_PATTERN.match(x).group(1)))
    # Optionally, sort incorrect files alphabetically
    incorrect.sort()

    # Determine starting index for renaming
    existing_nums = [int(PIC_PATTERN.match(x).group(1)) for x in correct]
    idx = max(existing_nums) + 1 if existing_nums else 1

    new_names = []
    # 1) Copy all correctly named files
    for fname in correct:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(images_out_dir, fname)
        shutil.copy2(src, dst)
        new_names.append(fname)

    # 2) Rename and copy incorrectly named files
    for fname in incorrect:
        src_old = os.path.join(input_dir, fname)
        ext = os.path.splitext(fname)[1].lower()
        new_name = f'pic_{idx:06d}{ext}'
        idx += 1

        src_new = os.path.join(input_dir, new_name)
        os.rename(src_old, src_new)

        dst = os.path.join(images_out_dir, new_name)
        shutil.copy2(src_new, dst)
        new_names.append(new_name)

    return new_names


def run_detection(client, image_path, prompt, model, api_path, bbox_thr, iou_thr):
    api_body = {
        "model": model,
        "prompt": {"type": "text", "text": prompt},
        "targets": ["bbox"],
        "bbox_threshold": bbox_thr,
        "iou_threshold": iou_thr
    }
    task = create_task_with_local_image_auto_resize(
        api_path=api_path,
        api_body_without_image=api_body,
        image_path=image_path
    )
    client.run_task(task)
    return task.result.get('objects', [])


def convert_to_yolo_format(detections, img_w, img_h,
                           default_thresh, category_thresholds,
                           class_to_idx):
    lines = []
    for obj in detections:
        score = obj.get('score', obj.get('confidence', 0))
        raw   = obj.get('category') or obj.get('text', '')
        label = normalize_label(raw)
        thr   = category_thresholds.get(label, default_thresh)
        if score < thr or label not in class_to_idx:
            continue
        cls_id = class_to_idx[label]
        x0, y0, x1, y1 = obj['bbox']
        cx = ((x0 + x1) / 2) / img_w
        cy = ((y0 + y1) / 2) / img_h
        bw = (x1 - x0) / img_w
        bh = (y1 - y0) / img_h
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def visualize_detections(detections, img_path, vis_path,
                         default_thresh, category_thresholds,
                         palette, color_map, font):
    """
    Draw bounding boxes and labels on the image, then save to vis_path.
    """
    image = Image.open(img_path).convert("RGB")
    draw  = ImageDraw.Draw(image)
    for obj in detections:
        score = obj.get('score', obj.get('confidence', 0))
        raw   = obj.get('category') or obj.get('text', '')
        label = normalize_label(raw)                # ← 归一化
        thr   = category_thresholds.get(label, default_thresh)
        if score < thr or label not in CLASS_TO_IDX:
            continue

        if label not in color_map:
            color_map[label] = palette[len(color_map) % len(palette)]
        box_color = color_map[label]
        x0, y0, x1, y1 = obj['bbox']
        draw.rectangle([x0, y0, x1, y1], outline=box_color, width=2)
        text = f"{label}: {score:.2f}"
        bbox_text = draw.textbbox((x0, y0), text, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        draw.rectangle([x0, y0 - th, x0 + tw, y0], fill=box_color)
        draw.text((x0, y0 - th), text, fill="white", font=font)

    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
    image.save(vis_path)


def main():
    args = parse_args()
    out_images = os.path.join(args.output_dir, 'train', 'images')
    out_labels = os.path.join(args.output_dir, 'train', 'labels')
    os.makedirs(out_labels, exist_ok=True)
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)

    # Copy and rename images
    new_images = rename_and_copy(args.input_dir, out_images)

    # Initialize API client
    config = Config(args.token)
    client = Client(config)

    prompt_text = args.prompt or '. '.join(CATEGORIES) + '.'

    # Generate deduplicated, normalized class names list
    names = []
    for cat in CATEGORIES:
        norm = normalize_label(cat)
        if norm not in names:
            names.append(norm)
    class_to_idx = {label: idx for idx, label in enumerate(names)}

    # Prepare color palette for visualization
    random.seed(42)
    palette = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]
    color_map = {}
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()

    # Process each image with progress bar
    num_exist_labels = 0  # Counter for images skipped due to existing labels
    for img_name in tqdm(new_images, desc="Processing images", unit="img"):
        # Skip if label already exists
        label_file = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(out_labels, label_file)
        if os.path.exists(label_path):
            num_exist_labels += 1
            continue

        img_path = os.path.join(out_images, img_name)
        detections = run_detection(
            client, img_path, prompt_text,
            args.model, args.api_path,
            args.bbox_threshold, args.iou_threshold
        )
        # YOLO labels
        with Image.open(img_path) as im:
            w, h = im.size
        yolo_lines = convert_to_yolo_format(
            detections, w, h,
            args.bbox_threshold,
            CATEGORY_THRESHOLDS,
            class_to_idx
        )
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        # Visualization
        if args.vis_dir:
            vis_path = os.path.join(args.vis_dir, img_name)
            visualize_detections(
                detections, img_path, vis_path,
                args.bbox_threshold,
                CATEGORY_THRESHOLDS,
                palette, color_map, font
            )

    # Write data.yaml
    data = {
        'train': 'train/images',
        'nc': len(names),
        'names': names
    }
    with open(os.path.join(args.output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"YOLO dataset has been created at: {args.output_dir}")
    print(f"Total images processed: {len(new_images)}")
    print(f"Existing images skipped: {num_exist_labels}")
    print(f"New images added: {len(new_images) - num_exist_labels}")


if __name__ == '__main__':
    main()


"""
Example usage:
python auto_annotate.py \
  --input-dir test_clothes_pics/ \
  --output-dir yolo_dataset/ \
  --vis-dir visualized_images/ \
  --token "YOUR_API_TOKEN"
"""