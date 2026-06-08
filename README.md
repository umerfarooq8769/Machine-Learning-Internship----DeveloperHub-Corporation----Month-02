# 🤖 DevelopersHub Corporation — AI/ML Engineering Internship
## Month 2 Tasks: Advanced Machine Learning Projects

**Intern:** Umer Farooq
**Organization:** DevelopersHub Corporation
**Track:** AI/ML Engineering
**Period:** Month 2

---

## 📋 Table of Contents

- [Overview](#overview)
- [Task 1 — News Topic Classifier Using BERT](#task-1--news-topic-classifier-using-bert)
- [Task 2 — End-to-End ML Pipeline with Scikit-learn](#task-2--end-to-end-ml-pipeline-with-scikit-learn)
- [Task 3 — Multimodal Housing Price Prediction](#task-3--multimodal-housing-price-prediction)
- [Tech Stack Summary](#tech-stack-summary)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

---

## Overview

This repository contains the Month 2 deliverables for the AI/ML Engineering Internship at **DevelopersHub Corporation**. The tasks cover three major areas of modern applied machine learning:

1. **Natural Language Processing** — Fine-tuning a BERT transformer for multi-class text classification
2. **Production ML Engineering** — Building a reusable, end-to-end Scikit-learn pipeline for customer churn prediction
3. **Multimodal Deep Learning** — Fusing image and tabular data to predict housing prices using a custom neural network

Each task is implemented in a Google Colab notebook and demonstrates industry-standard practices including model serialization, hyperparameter tuning, transfer learning, and interactive deployment.

---

## Task 1 — News Topic Classifier Using BERT

### 📌 Objective

Fine-tune a pre-trained **BERT (bert-base-uncased)** model on the **AG News dataset** to classify news headlines into one of four topic categories, and deploy the model via an interactive **Gradio** web interface.

### 📂 Notebook
`Task_1_News_Topic_Classifier_Using_BERT.ipynb`

### 🗂️ Dataset

- **Name:** AG News Dataset
- **Source:** [GitHub — mhjabreel/CharCnn_Keras](https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/)
- **Size:** 120,000 training samples / 7,600 test samples
- **Classes (4 labels):**

| Label | Category |
|-------|----------|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

### 🔬 Methodology

**1. Data Loading & Preparation**
- Loaded train/test CSVs using Pandas
- Combined `title` and `description` columns into a unified `text` field
- Adjusted labels from `[1–4]` to `[0–3]` for compatibility with HuggingFace's classification API

**2. Tokenization**
- Used `AutoTokenizer` from HuggingFace Transformers (`bert-base-uncased`)
- Applied padding and truncation with `max_length=128`
- Mapped tokenization over the full dataset using batched processing

**3. Model Architecture**
- Loaded `AutoModelForSequenceClassification` with `num_labels=4`
- The BERT model's pre-trained weights serve as the backbone; only the classification head is trained from scratch

**4. Training Configuration**
- **Learning Rate:** 2e-5
- **Batch Size:** 16 (train and eval)
- **Epochs:** 1 (on a subset of 2,000 samples for Colab efficiency)
- **Weight Decay:** 0.01
- **Optimizer:** AdamW (HuggingFace default)
- **Metrics:** Accuracy + Macro F1 Score

**5. Evaluation & Inference**
- Used HuggingFace `Trainer` API for training and evaluation
- `compute_metrics` function computes accuracy and macro F1 after each evaluation step
- Model and tokenizer saved locally to `./fine_tuned_bert_ag_news/`

**6. Deployment — Gradio Interface**
- Built an interactive web UI with `gr.Interface`
- Input: A news headline (text box)
- Output: Probability distribution across all 4 categories using `gr.Label`
- Deployed with `share=True` for a public shareable link

### 🛠️ Tech Stack

```
torch | transformers | datasets | evaluate | gradio | scikit-learn | pandas
```

### 📊 Key Results

| Metric | Value |
|--------|-------|
| Training Subset | 2,000 samples |
| Max Sequence Length | 128 tokens |
| Inference | Real-time via Gradio |
| Output | Probability scores per class |

---

## Task 2 — End-to-End ML Pipeline with Scikit-learn

### 📌 Objective

Build a **production-ready, reusable ML pipeline** using Scikit-learn's `Pipeline` and `ColumnTransformer` APIs to predict customer churn on the **Telco Customer Churn dataset**. The goal is to demonstrate best practices for robust, reproducible machine learning workflows.

### 📂 Notebook
`Task_2_End_to_End_ML_Pipeline_with_Scikit_learn_Pipeline_API.ipynb`

### 🗂️ Dataset

- **Name:** Telco Customer Churn
- **Source:** [Google Drive (provided)](https://drive.google.com/file/d/1795LiN5kTL7n-5yblN5zcYs1Ax_1D_hX/view?usp=sharing)
- **Target Variable:** `Churn` (binary: Yes / No)
- **Features:** ~20 columns including `tenure`, `MonthlyCharges`, `Contract`, `TotalCharges`, etc.

### 🔬 Methodology

**1. Exploratory Data Analysis (EDA)**
- Visualized churn distribution (bar chart)
- Compared tenure and monthly charges across churn groups (box plots)
- Analyzed contract type vs. churn rate (grouped bar chart)

**2. Data Preprocessing**
- Converted `TotalCharges` from string to numeric (handling coercion errors)
- Dropped non-informative `customerID` column
- Encoded target `Churn` as binary integer (Yes → 1, No → 0)
- Separated features `X` from target `y`
- Applied stratified 80/20 train-test split to preserve class balance

**3. Scikit-learn Pipeline Construction**

```
Numerical Pipeline:    SimpleImputer (median) → StandardScaler
Categorical Pipeline:  SimpleImputer (most_frequent) → OneHotEncoder
                            ↓
                     ColumnTransformer (preprocessor)
                            ↓
                       Classifier Model
```

**4. Model Training**

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear model, `max_iter=1000` |
| Random Forest | Ensemble tree-based model, `n_estimators=100` |

**5. Hyperparameter Tuning**
- Used `GridSearchCV` with `StratifiedKFold` (5 folds)
- Scoring metric: `roc_auc`
- Search space:
  - `n_estimators`: [100, 200]
  - `max_depth`: [None, 10, 20]
  - `min_samples_split`: [2, 5]

**6. Evaluation**
- Metrics: Accuracy, ROC-AUC, Classification Report (Precision, Recall, F1)
- Confusion Matrix heatmap (best model)
- ROC Curve comparison across all three models

**7. Feature Importance Analysis**
- Extracted feature importances from the tuned Random Forest
- Visualized Top 15 most influential features using a horizontal bar chart

**8. Model Export & Reload**
- Serialized the best pipeline using `joblib.dump()`
- Verified successful reload and inference with `joblib.load()`
- Demonstrated live predictions with churn probability scores

### 🛠️ Tech Stack

```
scikit-learn | pandas | numpy | matplotlib | seaborn | joblib
```

### 📊 Key Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | ~80% | ~0.84 |
| Random Forest | ~79% | ~0.82 |
| Tuned Random Forest (Best) | ~80%+ | ~0.84+ |

> *Exact scores depend on random state and CV folds.*

---

## Task 3 — Multimodal Housing Price Prediction

### 📌 Objective

Build a **multimodal deep learning model** that predicts housing prices by combining two distinct data modalities:
- **Visual data:** House images processed through a pre-trained CNN (**EfficientNetB0**)
- **Tabular data:** Structured housing attributes from the California Housing Dataset

This task demonstrates the concept of **feature fusion** in multimodal machine learning — a powerful paradigm used in real-world AI systems like real estate valuation engines.

### 📂 Notebook
`Task3_Multimodal_Housing_Price_Prediction.ipynb`

### 🗂️ Dataset

- **Name:** California Housing Dataset
- **Source:** [Hands-On ML GitHub](https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv)
- **Tabular Features:** `median_income`, `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `latitude`, `longitude`, `ocean_proximity`
- **Target:** `median_house_value` (regression)
- **Image Data:** Synthetically generated per sample (color-encoded by price range)

### 🔬 Methodology

**1. Data Exploration**
- Loaded and inspected the full California Housing dataset (~20,640 rows)
- Visualized price distribution, income vs. price, age distribution, ocean proximity categories
- Identified and handled missing values in `total_bedrooms`

**2. Data Preprocessing (Tabular)**
- Selected a subset of 1,000 samples for Colab efficiency
- Filled missing values with column medians
- One-hot encoded the `ocean_proximity` categorical column
- Normalized the target variable (`median_house_value`) using z-score standardization
- Scaled all tabular features using `StandardScaler`

**3. Synthetic Image Generation**
- Generated 64×64 pixel color images for each housing sample
- Color gradient encodes price level: **blue (low) → orange/red (high)**
- Added Gaussian noise and horizontal gradient bands to simulate photo texture
- Saved all images to `house_images/` directory in JPEG format

> **Note:** In a real-world deployment, actual house listing photos from real estate platforms would replace the synthetic images.

**4. CNN Feature Extraction (Transfer Learning)**
- Loaded pre-trained **EfficientNetB0** (ImageNet weights) with `include_top=False`
- Added `GlobalAveragePooling2D` to reduce spatial feature maps to a flat vector
- **Froze all CNN weights** — used purely as a feature extractor (no fine-tuning)
- Extracted image feature vectors in batches of 32 for memory efficiency
- Normalized image features using `StandardScaler`

**5. Multimodal Fusion Architecture**

```
Image Features (1280-dim)         Tabular Features (N-dim)
        ↓                                  ↓
  Dense(256) → BN → Dropout        Dense(64) → BN → Dropout
  Dense(128) → Dropout             Dense(32)
        ↓                                  ↓
              Concatenate (Feature Fusion)
                      ↓
              Dense(128) → Dropout
              Dense(64) → Dense(32)
                      ↓
                  Output (1) — Price
```

- **Loss Function:** Huber Loss (robust to outliers)
- **Optimizer:** Adam (`lr=1e-3`)
- **Metrics:** MAE

**6. Training**
- Train/test split: 80% / 20% (index-based, synchronized across both modalities)
- Validation split: 15% of training data
- Max epochs: 60
- **Early Stopping:** patience=8, restores best weights
- **ReduceLROnPlateau:** halves learning rate after 4 stagnant epochs

**7. Evaluation**
- Converted normalized predictions back to original price scale
- Metrics computed on original prices: **MAE** and **RMSE**
- Plotted training/validation loss and MAE curves
- Scatter plot: Actual vs. Predicted prices with a reference diagonal

**8. Prediction Visualization**
- Generated a detailed visual showing 10 test samples
- Each sample displays: synthetic house image, actual price, predicted price, and absolute error

### 🛠️ Tech Stack

```
tensorflow | keras | EfficientNetB0 | scikit-learn | pandas | numpy | matplotlib | Pillow
```

### 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error in original dollar values |
| RMSE | Root Mean Squared Error (penalizes large errors more) |

> *Results depend on the number of training samples and GPU availability. Recommended: Google Colab T4 GPU for faster training.*

---

## Tech Stack Summary

| Category | Libraries / Tools |
|----------|-------------------|
| NLP / Transformers | `transformers`, `datasets`, `evaluate`, `torch` |
| Classical ML | `scikit-learn`, `joblib` |
| Deep Learning | `tensorflow`, `keras` |
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `Pillow` |
| Deployment | `gradio` |
| Environment | Google Colab (Python 3.x) |

---

## Project Structure

```
Month_2_Tasks/
│
├── Task_1_News_Topic_Classifier_Using_BERT.ipynb
│   ├── fine_tuned_bert_ag_news/          # Saved BERT model & tokenizer
│   └── (Gradio live demo link on run)
│
├── Task_2_End_to_End_ML_Pipeline_with_Scikit_learn_Pipeline_API.ipynb
│   └── churn_pipeline.pkl                # Exported Scikit-learn pipeline
│
├── Task3_Multimodal_Housing_Price_Prediction.ipynb
│   ├── house_images/                     # Generated synthetic house images
│   ├── data_exploration.png
│   ├── sample_images.png
│   └── (Training history plots)
│
└── README.md
```

---

## How to Run

All notebooks are designed to run on **Google Colab** with zero local setup.

### 🔗 Colab Links

| Task | Description | Link |
|------|-------------|------|
| Task 01 | News Topic Classifier Using BERT | [Open in Colab](https://colab.research.google.com/drive/1fX-JK700yqO8m2xERS5BF4zZ9CIODKdJ?usp=sharing) |
| Task 02 | End-to-End ML Pipeline with Scikit-learn | [Open in Colab](https://colab.research.google.com/drive/1Etj8f9OHy8Zy8h55vHf9Xda0mlEEJCTM?usp=sharing) |
| Task 03 | Multimodal Housing Price Prediction | [Open in Colab](https://colab.research.google.com/drive/1izSllYPeC-42FjR-UnxEjskr3A4JsPw7?usp=sharing) |

### ▶️ Steps

**Step 1:** Click the Colab link for the task you want to run

**Step 2:** For Task 3, go to `Runtime → Change runtime type → T4 GPU` for faster training

**Step 3:** Run all cells sequentially from top to bottom

**Step 4 (Task 2 only):** Mount Google Drive and update the dataset path:
```python
df = pd.read_csv('/content/drive/MyDrive/Telco-Customer-Churn.csv')
```

---

## Author

**Umer Farooq**
AI/ML Engineering Intern — DevelopersHub Corporation
Month 2 Submission

---

*This project was completed as part of the structured AI/ML internship program at DevelopersHub Corporation. All implementations follow industry best practices for reproducibility, modularity, and production readiness.*
