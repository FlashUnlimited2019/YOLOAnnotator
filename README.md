# YOLOAnnotator: GroundingDINO-Powered YOLO Dataset & Visualization Pipeline

A lightweight Python tool to:

1. **Normalize & rename** your raw images into a consistent `pic_000001.jpg` format.
2. **Batch-infer** object detections with the GroundingDINO-1.6-Pro API.
3. **Export YOLO-formatted labels** (`.txt`) under `train/labels/`.
4. **Generate a `data.yaml`** file for your YOLO training setup.
5. **(Optional) Produce visualizations** with colored bounding-boxes, class labels & confidences.
6. **Skip already-annotated images** to avoid redundant API calls.
7. **Support per-category confidence thresholds** (e.g. higher bar for “camera”).
8. **Unify synonyms** in detections (e.g. `T-shirt` → `shirt`) so your label set stays clean.
9. **Show a progress bar** for long runs via `tqdm`.

---

## 🔧 Requirements

- Python 3.8+
- [`dds-cloudapi-sdk`](https://github.com/deepdataspace/dds-cloudapi-sdk)
- `Pillow`
- `PyYAML`
- `tqdm`

Install with:

```bash
pip install dds-cloudapi-sdk pillow pyyaml tqdm
```

---

## 🛠 Usage

```bash
python auto_annotate.py \
  --input-dir  /path/to/raw_images \
  --output-dir /path/to/yolo_dataset \
  --vis-dir    /path/to/visualizations \      # optional
  --token      YOUR_DDS_API_TOKEN            # required
  [--prompt   "handbag. dress. shoes. ..."]  # optional custom prompt
```

### Command-line Arguments

| Argument           | Description                                                                     | Default                             |
| ------------------ | ------------------------------------------------------------------------------- | ----------------------------------- |
| `--input-dir`      | Directory containing your raw `.jpg` / `.png` images                            | ― (required)                        |
| `--output-dir`     | Root of the output YOLO dataset (will create `train/images`, `train/labels`)    | ― (required)                        |
| `--vis-dir`        | Where to save colored visualizations (same filenames as inputs)                 | `None` (no visualizations)          |
| `--token`          | Your DDS Cloud API token                                                        | ― (required)                        |
| `--prompt`         | Custom text prompt for GroundingDINO (defaults to all categories joined by “.”) | Combined `CATEGORIES` list          |
| `--model`          | GroundingDINO model name                                                        | `GroundingDino-1.6-Pro`             |
| `--api-path`       | DDS API endpoint path for detection                                             | `/v2/task/grounding_dino/detection` |
| `--bbox-threshold` | Default confidence threshold (0–1)                                              | `0.3`                               |
| `--iou-threshold`  | API-side NMS IoU threshold (0 to disable)                                       | `0.8`                               |

---

## 📁 Output Structure

After running, your `output-dir` will contain:

```
output-dir/
├── data.yaml            # YOLO data config (train/images, nc, names)
└── train/
    ├── images/          # Renamed “pic_000001.jpg”, “pic_000002.png”, …
    └── labels/          # YOLO .txt files with “class x_center y_center w h”
```

If `--vis-dir` is specified:

```
visualizations/
└── pic_000001.jpg       # Same filename but with colored boxes & labels
```

---

## ⚙️ Advanced Features

* **Skip duplicates**: If `train/labels/pic_000123.txt` already exists, that image is *not* re-inferred.
* **Per-class thresholds**: Tweak `CATEGORY_THRESHOLDS` in the script to require higher confidence for select classes.
* **Label unification**: Use `LABEL_UNIFY` to merge synonyms (e.g. map `t-shirt`, `Tshirt` → `shirt`).
* **Consistent colors**: Each class gets a unique RGB color for visualization; same class = same color.

---

## ✨ Example

```bash
python auto_annotate.py \
  --input-dir ./photos/ \
  --output-dir ./my_yolo_data/ \
  --vis-dir ./vis/ \
  --token "xxxxx"
```

---
