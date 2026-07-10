# Dataset Information — Banknote Authentication

## Source

- **Name:** Banknote Authentication Dataset
- **Provider:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/dataset/267/banknote+authentication
- **File used:** `data_banknote_authentication.txt`

## Description

The dataset was created by extracting statistical features from wavelet-transformed images of genuine and forged banknote specimens. Images were captured using an industrial camera used for print inspection, at 400x400 pixel resolution, then converted to grayscale (~660 dpi). The task is to classify a banknote as **authentic (genuine)** or **forged (fake)** based on these extracted features.

## Basic Statistics

| Property | Value |
|---|---|
| Number of instances | 1,372 |
| Number of features | 4 (all continuous, numeric) |
| Target classes | 2 (binary classification) |
| Missing values | None |
| File format | CSV-like, no header row, comma-separated |

## Features / Columns

| Column | Description | Type |
|---|---|---|
| `variance` | Variance of wavelet-transformed image | float |
| `skewness` | Skewness of wavelet-transformed image | float |
| `curtosis` | Kurtosis of wavelet-transformed image | float |
| `entropy` | Entropy of the image | float |
| `class` | Target label — `0` = authentic, `1` = forged | int |

> Note: the raw file has **no header row**; the notebook assigns these column names manually via `pandas.read_csv(..., header=None, names=COLUMNS)`.

## How to Download

1. Visit https://archive.ics.uci.edu/dataset/267/banknote+authentication
2. Download the dataset (the file is typically named `data_banknote_authentication.txt`, sometimes packaged inside a `.zip`).
3. Place the `.txt` file in the project's `data/` folder (or the same folder as the notebook), matching the `DATA_PATH` variable used in the code:
   ```python
   DATA_PATH = "data_banknote_authentication.txt"
   ```

## License / Usage

This dataset is publicly available through the UCI Machine Learning Repository for research and educational use. Please refer to the UCI page above for citation details if required by your course/report.
