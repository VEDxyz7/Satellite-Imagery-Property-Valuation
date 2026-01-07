# Satellite Imagery–Based Property Valuation

- Name : Ved Parikh
- Enrollment No : 24112080
- Project Report : [Satellite Imagery Based Property Valuation Report.pdf](https://github.com/user-attachments/files/24468300/Satellite.Imagery.Based.Property.Valuation.Report.pdf)


## Project Overview

This project implements a multimodal regression pipeline to predict residential property prices by combining:

- Structured tabular housing data (size, location, quality, neighborhood statistics)
- Satellite imagery capturing environmental and spatial context

The objective is to enhance traditional real-estate valuation models by incorporating visual cues such as greenery, urban density, road connectivity, and surrounding infrastructure, while also ensuring model explainability using Grad-CAM.

---

## Objectives

- Build a multimodal regression model for property price prediction  
- Programmatically fetch satellite images using latitude–longitude coordinates  
- Perform exploratory data analysis (EDA) and geospatial analysis  
- Extract visual features using a convolutional neural network (ResNet18)  
- Fuse image embeddings with tabular features  
- Provide visual explainability via Grad-CAM  
- Generate predictions for the unseen test dataset in the required CSV format  

---

## Dataset Description

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

- Satellite images fetched using the Mapbox Static Images API  
- Images centered at property latitude–longitude coordinates  
- Used to capture environmental and neighborhood context  

---

## Modeling Approach

### Tabular Baseline Model

- Random Forest Regressor  
- Trained on structured features only  
- Used as:
  - A performance baseline  
  - A fallback model during inference when satellite images are unavailable  

Validation performance:
- RMSE ≈ 145,000  
- R² ≈ 0.83  

---

### Multimodal Model

- Image Encoder: ResNet18 (ImageNet pretrained, frozen)  
- Tabular Encoder: Multi-layer perceptron (MLP)  
- Fusion Strategy: Late fusion via feature concatenation  
- Loss Function: Mean Squared Error (log-price space)  

This architecture allows the model to jointly learn from numeric and visual inputs.

---

## Explainability (Grad-CAM)

Grad-CAM is applied to the final convolutional layer of the image encoder to visualize regions in satellite imagery that influence model predictions.

Observed focus areas include:
- Vegetation and green cover  
- Urban density patterns  
- Surrounding infrastructure and road networks  

These visualizations help interpret how environmental context contributes to property valuation.

---

## Inference Strategy (Hybrid)

To ensure stable predictions for all test samples:

- If a satellite image exists, the multimodal model is used  
- If a satellite image is missing or unreliable, a tabular-only fallback model is used  

This hybrid strategy improves robustness and reflects real-world deployment constraints.

---

## Submission Artifacts

- Code Repository: Public GitHub repository  
- Prediction File: `24112080_final.csv`  
  - Format: `id, predicted_price`  
- Final Report: `24112080_report.pdf`  

---

## Tech Stack

- Data Handling: Pandas, NumPy  
- Deep Learning: PyTorch  
- Image Processing: PIL  
- Machine Learning: Scikit-learn  
- Visualization: Matplotlib  
 
- **Image Processing:** PIL  
- **Machine Learning:** Scikit-learn  
- **Visualization:** Matplotlib  

---
