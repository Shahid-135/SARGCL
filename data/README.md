# Data Preparation

## CrisisMMD Dataset

This project uses the **CrisisMMD** multimodal crisis-informatics dataset. The dataset contains tweets with images and text from real-world disaster events, labeled as **informative** or **not_informative**.

### Download

1. Download the CrisisMMD dataset from the official source:
   - **Paper**: [CrisisMMD: Multimodal Twitter Datasets from Natural Disasters](https://arxiv.org/abs/1805.00713)
   - **Download link**: [https://crisisnlp.qcri.org/crisismmd](https://crisisnlp.qcri.org/crisismmd)

2. Extract the dataset and note the paths to:
   - The **image directory** (e.g., `data_image/`)
   - The **TSV annotation files**:
     - `task_informative_text_img_agreed_lab_train.tsv`
     - `task_informative_text_img_agreed_lab_dev.tsv`
     - `task_informative_text_img_agreed_lab_test.tsv`

### Expected Directory Structure

```
data/
├── README.md                                          (this file)
├── task_informative_text_img_agreed_lab_train.tsv      (training split)
├── task_informative_text_img_agreed_lab_dev.tsv        (validation split)
├── task_informative_text_img_agreed_lab_test.tsv       (test split)
└── data_image/                                        (tweet images)
    ├── california_wildfires/
    ├── hurricane_harvey/
    ├── hurricane_irma/
    ├── hurricane_maria/
    ├── iraq_iran_earthquake/
    ├── mexico_earthquake/
    ├── srilanka_floods/
    └── ...
```

### TSV File Format

Each TSV file contains the following columns:

| Column | Description |
|--------|-------------|
| `event_name` | Name of the disaster event |
| `tweet_id` | Unique tweet identifier |
| `image_id` | Image identifier |
| `tweet_text` | Raw tweet text |
| `image` | Relative path to the image file |
| `label` | Classification label: `informative` or `not_informative` |
| `label_text` | Human-readable label description |
| `label_image` | Image-based label |

### Usage

Once the data is in place, you can train with:

```bash
python scripts/train.py \
    --train_csv data/task_informative_text_img_agreed_lab_train.tsv \
    --valid_csv data/task_informative_text_img_agreed_lab_dev.tsv \
    --test_csv  data/task_informative_text_img_agreed_lab_test.tsv \
    --image_dir data/data_image
```
