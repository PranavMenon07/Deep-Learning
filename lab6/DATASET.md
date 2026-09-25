# Dataset Description

This experiment uses two real-world datasets and one synthetically generated dataset. None of
the raw data is committed to this repository (see `.gitignore`) — each must be downloaded or
generated locally before running the notebook.

---

## 1. UCI Human Activity Recognition Using Smartphones (primary dataset)

**Used for:** Sections on RNN/LSTM/GRU sequence classification, temporal visualization,
sequence-length study.

- **Source:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Content:** Raw inertial signals from a waist-mounted smartphone (accelerometer + gyroscope),
  collected from 30 subjects performing six activities.
- **Classes (6):** `WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`,
  `LAYING`
- **Signal format:** each activity window = 128 time steps × 9 channels (body acceleration x/y/z,
  gyroscope x/y/z, total acceleration x/y/z), i.e. `X ∈ R^(N × 128 × 9)`.
- **Important:** the notebook uses the **raw `Inertial Signals/` files**, not the pre-computed
  561-feature vectors (`X_train.txt` / `X_test.txt` at the dataset root).
- **Download:** handled automatically by the notebook (`urllib.request`), or manually:
  ```bash
  wget https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip
  unzip "human+activity+recognition+using+smartphones.zip"
  unzip "UCI HAR Dataset.zip"
  ```
- **Expected local layout** (matches `DATA_DIR` in the notebook):
  ```
  UCI_HAR_Dataset/
      train/
          Inertial Signals/
              body_acc_x_train.txt ... total_acc_z_train.txt
          y_train.txt
      test/
          Inertial Signals/
              body_acc_x_test.txt ... total_acc_z_test.txt
          y_test.txt
  ```
- **License / citation:** Anguita et al., "A Public Domain Dataset for Human Activity
  Recognition Using Smartphones," ESANN 2013. Public domain / free for research use per the
  UCI ML Repository terms.

### Laboratory subset
To keep training time small, a balanced subset of **1500–3000 windows** (preserving all six
classes, approximately equal per class) is drawn from the combined train+test pool, then
re-split **70% / 15% / 15%** into train / validation / test. The test split is never used for
model selection or hyperparameter tuning.

---

## 2. UCF101 Action Recognition (video understanding subset)

**Used for:** the CNN (MobileNetV2) + LSTM/GRU video-understanding section (Plots 7–9).

- **Source:** https://www.crcv.ucf.edu/data/UCF101.php
- **Content:** 13,320 short video clips across 101 action classes. This experiment only needs a
  **small subset**: 3–5 classes, a handful of videos per class.
- **Recommended classes:** `Basketball`, `Biking`, `Walking`, `Running`, `TennisSwing` (any
  3–5 distinct, visually separable classes from the full list work).
- **Download (full archive is large, ~6.5 GB):**
  ```bash
  wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
  unrar x UCF101.rar
  ```
  then copy only the needed class folders into the subset directory below. Downloading the full
  archive is optional — any small set of short clips (including your own recordings) belonging
  to the chosen classes works for demonstrating the pipeline.
- **Expected local layout** (matches `VIDEO_ROOT` in the notebook):
  ```
  UCF101_subset/
      Basketball/*.avi   (or .mp4)
      Biking/*.avi
      Walking/*.avi
      Running/*.avi
      TennisSwing/*.avi
  ```
- **Minimum size for the notebook to run:** more than 5 videos total, with enough per class for
  a stratified 70/15/15-style split (≥ 6–10 videos per class is recommended so the split doesn't
  fail on small classes — see the notebook's `safe_split` helper for a fallback when data is
  very limited).
- **Not tracked in git:** place this folder outside the repo, or under a `data/` directory
  (already excluded via `.gitignore`), or mount it from Google Drive when using Colab.
- **License / citation:** Soomro, Zamir, Shah, "UCF101: A Dataset of 101 Human Actions Classes
  From Videos in the Wild," 2012. Free for research/academic use per the dataset's terms.

---

## 3. Synthetic Sequence Reversal Dataset (generated, no download needed)

**Used for:** the sequence-to-sequence (encoder–decoder LSTM) section.

- **Generation:** integers sampled uniformly from a fixed range (default `0–9`), fixed-length
  sequences (default length 4), with the target defined as the reversed input, e.g.
  `[3, 8, 1, 5] → [5, 1, 8, 3]`.
- **Size:** 5,000 samples generated in-notebook (`NUM_SAMPLES = 5000`), split 72% / 8% / 20%
  (train / val / test) via `train_test_split` calls.
- **Reproducibility:** generated with a fixed NumPy `default_rng(42)` seed — identical sequences
  are produced on every run.
- No external download or storage required; nothing from this dataset needs to be committed.

---

## Data Integrity Notes

- Preprocessing statistics (mean/std for normalization) are computed **only from the training
  split** and reused for validation/test — this is done in the notebook and must not be changed.
- The HAR test set and the video test split are held out from all model-selection and
  hyperparameter-tuning decisions, per the experiment protocol.
- Raw data files, extracted archives, and any downloaded `.zip`/`.rar`/`.avi`/`.mp4` files are
  excluded from version control by `.gitignore` — only code, the notebook, and the report are
  tracked.
