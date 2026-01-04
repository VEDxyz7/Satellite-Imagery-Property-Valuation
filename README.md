# 📡 Satellite Imagery–Based Property Valuation  
**CDC × Yhills Open Projects (2025–2026)**  
**Enrollment Number:** 24112080  

---

## 📌 Project Overview

This project implements a **multimodal regression pipeline** to predict residential property prices by combining:

- **Structured tabular housing data** (size, location, quality, neighborhood statistics)
- **Satellite imagery** capturing environmental and spatial context

The motivation is to enhance traditional real-estate valuation models by incorporating visual cues such as greenery, urban density, road connectivity, and surrounding infrastructure, while also ensuring **model explainability** using Grad-CAM.

---

## 🎯 Objectives

- Build a **multimodal regression model** for property price prediction  
- Programmatically fetch satellite images using latitude–longitude coordinates  
- Perform **exploratory data analysis (EDA)** and geospatial analysis  
- Extract visual features using a **CNN (ResNet18)**  
- Fuse image embeddings with tabular features  
- Provide **visual explainability** via Grad-CAM  
- Generate predictions for the unseen test dataset in the required CSV format  

---

## 🗂️ Repository Structure

PropertyValuation/
│
├── data/
│ ├── raw/
│ │ ├── train.csv
│ │ └── test.csv
│ └── images/ # Satellite images (not pushed to GitHub)
│
├── notebooks/
│ ├── preprocessing.ipynb # EDA and preprocessing
│ ├── model_training.ipynb # Model training, evaluation, inference
│
├── src/
│ ├── data_fetcher.py # Satellite image downloader (Mapbox API)
│ ├── dataset.py # Custom PyTorch Dataset
│ └── model.py # Multimodal model architecture
│
├── outputs/
│ └── 24112080_final.csv # Final test predictions (submission file)
│
├── README.md
├── requirements.txt
└── .gitignore


---

## 📊 Dataset Description

### Tabular Data
Key features used:
- `bedrooms`, `bathrooms`
- `sqft_living`, `sqft_lot`
- `grade`, `condition`
- `lat`, `long`
- Neighborhood statistics (`sqft_living15`, `sqft_lot15`)

Target variable:
- `price` (log-transformed during training)

### Visual Data
- Satellite images fetched using **Mapbox Static Images API**
- Images centered at property latitude–longitude coordinates
- Used to capture environmental and neighborhood context

---

## 🧠 Modeling Approach

### 1️⃣ Tabular Baseline Model
- Random Forest Regressor
- Trained on structured features only
- Used as:
  - Performance baseline
  - Fallback model during inference when images are unavailable

**Validation Performance:**
- RMSE ≈ 145,000  
- R² ≈ 0.83  

---

### 2️⃣ Multimodal Model
- **Image Encoder:** ResNet18 (ImageNet pretrained, frozen)
- **Tabular Encoder:** Multi-layer perceptron (MLP)
- **Fusion Strategy:** Late fusion via feature concatenation
- **Loss Function:** Mean Squared Error (log-price space)

This architecture allows the model to integrate both numeric and visual signals.

---

## 🔍 Explainability (Grad-CAM)

Grad-CAM was applied to the final convolutional layer of the image encoder to visualize regions in satellite imagery that influenced predictions.

Observed focus areas include:
- Vegetation and green cover
- Urban density patterns
- Surrounding infrastructure and road networks

These visualizations help interpret how environmental context contributes to property valuation.

---

## 📈 Evaluation Strategy

- Model performance evaluated using a **train–validation split** on `train.csv`
- Metrics reported:
  - RMSE
  - R²
  - Log-RMSE (for multimodal model)
- Final predictions on `test.csv` were generated **without access to ground-truth labels**, consistent with standard supervised learning practice

---

## 🧪 Inference Strategy (Hybrid)

To ensure predictions for all test samples:

- **If satellite image exists:**  
  → Multimodal model is used  
- **If satellite image is missing:**  
  → Tabular-only fallback model is used  

This hybrid approach ensures robustness and reflects real-world deployment scenarios.

---

## 📤 Submission Artifacts

- **Code Repository:** Public GitHub repository  
- **Prediction File:** `24112080_final.csv`  
  - Format: `id, predicted_price`  
- **Final Report:** `24112080_report.pdf` (PDF only)  

---

## ⚠️ Limitations

- Satellite imagery was fetched for a limited subset of properties due to API constraints  
- Multimodal model performance is limited by the number of available images  
- The primary focus of this project is pipeline design, explainability, and engineering quality rather than absolute price calibration  

---

## 🛠️ Tech Stack

- **Data Handling:** Pandas, NumPy  
- **Deep Learning:** PyTorch  
- **Image Processing:** PIL  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Matplotlib  

---