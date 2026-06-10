# Fake News Detection using Deep Learning

A comparative study of deep learning models for fake news detection on the ISOT Fake News Dataset. This project was developed as a Final Project for a Deep Learning course.

---

## Overview

This project compares four deep learning architectures for binary fake news classification:

| Model | Accuracy | F1 | AUC-ROC | Train Time |
|---|---|---|---|---|
| LSTM | 0.9900 | 0.9904 | 0.9991 | 178.3s |
| BiLSTM | 0.9962 | 0.9964 | 0.9998 | 297.1s |
| TextCNN | 0.9982 | 0.9983 | 1.0000 | 82.0s |
| DistilBERT | 0.9987 | 0.9987 | 1.0000 | 5509.8s |

**Key finding:** TextCNN achieves near-identical performance to DistilBERT while being **67× faster** to train, making it the most practical model for this task.

---

## Dataset

**ISOT Fake News Dataset** — University of Victoria

- 44,898 articles total (21,417 real + 23,481 fake)
- Source: Reuters (real) and various unreliable sources (fake)
- Features: `title`, `text`, `subject`, `date`
- Labels: Binary (0 = Real, 1 = Fake)
- Download: [Kaggle — ISOT Fake News Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

> **Note:** The dataset is not included in this repository due to licensing. Please download it from Kaggle and place `True.csv` and `Fake.csv` in the `data/` folder.

---

## Repository Structure

```
fake-news-detection/
│
├── fake_news_detection.ipynb   # Main notebook (EDA → Preprocessing → Training → Evaluation)
│
├── results/                    # All generated plots and figures
│   ├── class_distribution.png
│   ├── wordcloud_real.png
│   ├── wordcloud_fake.png
│   ├── lstm_training_curves.png
│   ├── bilstm_training_curves.png
│   ├── textcnn_training_curves.png
│   ├── distilbert_training_curves.png
│   ├── all_confusion_matrices.png
│   ├── roc_curve_comparison.png
│   ├── scenario1_model_comparison.png
│   ├── scenario2_input_ablation.png
│   └── scenario3_preprocessing_impact.png
│
├── data/
│   └── README.md           
│
└── README.md
```

---

## Methodology

### Preprocessing Pipeline
1. Merge `True.csv` and `Fake.csv` with binary labels
2. Drop non-text columns (`date`, `subject`)
3. Expand contractions (`don't` → `do not`)
4. Remove URLs, HTML tags, Reuters bylines, punctuation
5. Remove single-character noise tokens
6. Combine title + body with `[SEP]` separator
7. Stratified train/val/test split — 80% / 10% / 10%

### Models
| Model | Architecture | Embedding |
|---|---|---|
| LSTM | Single LSTM (128 units) + Dropout | GloVe 100d |
| BiLSTM | Bidirectional LSTM (128×2 units) + Dropout | GloVe 100d |
| TextCNN | Parallel Conv1D (kernel 2,3,4) + GlobalMaxPool | GloVe 100d |
| DistilBERT | Fine-tuned DistilBERT + Linear head | Pretrained |

### Experimental Scenarios
| Scenario | Description | Model Used |
|---|---|---|
| Scenario 1 | Model comparison | All four models |
| Scenario 2 | Input ablation: Title only / Body only / Title+Body | TextCNN |
| Scenario 3 | Preprocessing impact: No clean / Basic / Full pipeline | TextCNN |

---

## Results

### Scenario 2 — Input Feature Ablation

| Input | Accuracy | F1 |
|---|---|---|
| Title Only | 0.9592 | 0.9609 |
| Body Only | 0.9967 | 0.9968 |
| Title + Body | 0.9982 | 0.9983 |

Article body carries the strongest signal. Combining title and body consistently yields the best performance.

### Scenario 3 — Preprocessing Impact

| Preprocessing | Accuracy | F1 |
|---|---|---|
| No Cleaning | 0.9993 | 0.9994 |
| Basic Cleaning | 0.9984 | 0.9985 |
| Full Pipeline | 0.9980 | 0.9981 |

Performance differences are minimal (<0.002) on ISOT due to its high data quality. Full pipeline is still recommended for robustness on noisier datasets.

---

## How to Run

### Requirements
```bash
pip install tensorflow transformers torch pandas numpy scikit-learn matplotlib seaborn wordcloud tqdm
```

### Steps
1. Clone this repository
```bash
git clone https://github.com/k08e24bryant/Fake-News-Detection-using-Deep-Learning.git
cd Fake-News-Detection-using-Deep-Learning
```

2. Download the ISOT dataset from Kaggle and place files in `data/`:
```
data/
├── True.csv
└── Fake.csv
```

3. Open `fake_news_detection.ipynb` in Google Colab or Jupyter

4. Update the file paths in the config cells to match your environment

5. Run all cells in order from top to bottom

---

## Environment

- Python 3.10+
- TensorFlow 2.x
- PyTorch 2.x (DistilBERT only)
- HuggingFace Transformers 4.41+
- Google Colab (T4 GPU)
- GloVe embeddings: `glove.6B.100d` ([Download](https://nlp.stanford.edu/projects/glove/))

---

## Evaluation Metrics

| Metric | Formula | Why Used |
|---|---|---|
| Accuracy | (TP+TN) / Total | Overall correctness |
| Precision | TP / (TP+FP) | How many flagged articles are actually fake |
| Recall | TP / (TP+FN) | How many fake articles are caught |
| F1-Score | 2 × (P×R)/(P+R) | Balance between Precision and Recall |
| AUC-ROC | Area under ROC curve | Model's ability to discriminate classes |

---

## Author

**Syarif**  
Undergraduate Student — Teknik Informatika  
Institut Teknologi Sepuluh Nopember (ITS) Surabaya

---


