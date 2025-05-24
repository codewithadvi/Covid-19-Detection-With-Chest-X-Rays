#  COVID-19 Chest X-Ray Classification using HOG and Machine Learning

This project aims to automatically **classify chest X-ray images** into two categories—**COVID-19 infected** and **Normal**—using a combination of **handcrafted features (HOG)** and traditional **machine learning algorithms** such as **Support Vector Machine (SVM)**, **Random Forest**, and **XGBoost**.

Unlike deep learning-based approaches, this solution demonstrates the effectiveness of classical computer vision + ML pipelines, which are computationally lightweight and do not require large-scale GPU infrastructure.

---

##  Motivation

During the COVID-19 pandemic, medical imaging such as chest X-rays became a crucial tool for rapid screening and diagnosis. While deep learning methods can provide high accuracy, they require large annotated datasets and high computational power—resources that may not be readily available in every medical facility.

This project explores an **alternative lightweight solution** using:
- **HOG (Histogram of Oriented Gradients)** for image feature extraction
- **Traditional machine learning classifiers** for prediction

This approach is interpretable, fast, and efficient, making it ideal for real-time or low-resource environments.

---

##  Dataset

- **Source:** [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Used Subsets:**
  - `COVID`: Chest X-rays of patients infected with COVID-19
  - `Normal`: X-rays of healthy individuals

The dataset contains high-resolution grayscale images. Only the **COVID** and **Normal** classes are used in this experiment to perform binary classification.

---

## Problem Statement

> **Objective:** To build an efficient binary classifier that can distinguish between COVID-infected and Normal chest X-rays with high accuracy using traditional machine learning.

The classifier should:
- Handle grayscale images
- Extract meaningful visual features
- Output accurate labels based on non-deep-learning techniques

---

##  Feature Extraction: HOG (Histogram of Oriented Gradients)

###  Why HOG?

HOG is an effective and lightweight technique that captures **edge structures**, **shapes**, and **gradients** in localized image patches. These are essential features in medical imaging, where abnormal lung textures or opacities can indicate disease.

### 🔬 How HOG Works

1. **Convert to Grayscale:** Simplifies analysis by removing color dimensions
2. **Resize to 128×128:** Standardizes image dimensions for feature extraction
3. **Gradient Computation:** Calculates edge directions in small regions
4. **Cell Histograms:** For each 8×8 pixel cell, computes a histogram of gradient directions (9 orientations)
5. **Block Normalization:** Groups of 2×2 cells are normalized to account for contrast differences
6. **Feature Vector Construction:** All block histograms are concatenated into a single feature vector

   <img width="746" alt="image" src="https://github.com/user-attachments/assets/92d83188-68c3-46d0-9eea-7ce0bfbeabc8" />

<img width="742" alt="image" src="https://github.com/user-attachments/assets/b906511d-6ac5-4c12-aaa8-eabd71f72487" />

### HOG Parameters Used

| Parameter           | Value         |
|---------------------|---------------|
| Image Size          | 128 × 128     |
| Orientations        | 9             |
| Pixels per Cell     | (8, 8)        |
| Cells per Block     | (2, 2)        |
| Block Normalization | `L2-Hys`      |

##  Machine Learning Models

After feature extraction, the dataset is split into **training (80%)** and **testing (20%)** sets. Feature scaling is applied using `StandardScaler` to ensure all features are on the same scale.

###  Models Trained

- **Support Vector Machine (SVM)** with RBF Kernel

<img width="865" alt="image" src="https://github.com/user-attachments/assets/36cae3c4-6536-46c2-a122-2a8bf5782990" />

<img width="518" alt="image" src="https://github.com/user-attachments/assets/dab0adb6-4d22-4f2c-b7b5-113ad09e88ee" />

- **Random Forest Classifier**

<img width="865" alt="image" src="https://github.com/user-attachments/assets/bf8a448d-8ce8-475e-a0ac-4fc0bc2a1f49" />

<img width="521" alt="image" src="https://github.com/user-attachments/assets/64982e42-fe8c-4969-b69e-00038e99ce05" />


- **XGBoost Classifier**

<img width="868" alt="image" src="https://github.com/user-attachments/assets/9324b3ad-57e3-48c1-ad96-059bb668fe9e" />

<img width="520" alt="image" src="https://github.com/user-attachments/assets/2a5b1e74-9ab7-49bb-b218-a95572b9cf71" />

These models were selected for their:
- Robustness to overfitting
- Generalization performance
- Effectiveness on small- to medium-sized datasets
- Ease of interpretability and tuning

---

## Evaluation Metrics

The following metrics were used to evaluate model performance:
- **Confusion Matrix**
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

---

### ✅ Results Summary

| Model                  | Accuracy (%) |
|------------------------|--------------|
| Support Vector Machine | **97.30**    |
| Random Forest          | 95.67        |
| XGBoost                | 96.10        |

<img width="593" alt="image" src="https://github.com/user-attachments/assets/5333c9a6-93f4-42f1-a6be-0e918116e72d" />

<img width="590" alt="image" src="https://github.com/user-attachments/assets/b5016b70-6f60-46e6-8f3a-70ceecf39bcf" />



> **SVM achieved the highest accuracy**, indicating strong performance in distinguishing between COVID-19 and Normal chest X-rays using only handcrafted features from HOG.

##  Sample Classification Report (SVM)

              precision    recall  f1-score   support

       COVID       0.97      0.98      0.97       100
      Normal       0.98      0.97      0.97       100

    accuracy                           0.97       200
   macro avg       0.97      0.97      0.97       200
weighted avg       0.97      0.97      0.97       200


---

## Highlights

- No deep learning or GPUs required
- Lightweight and fast execution
- Traditional ML model interpretability
- Accuracy exceeding **97%** with handcrafted features

---

##  Future Improvements

- Include other categories such as **Viral Pneumonia** and **Lung Opacity**
- Experiment with deep learning models like **MobileNet** or **EfficientNet**
- Apply **PCA** or **t-SNE** for feature space visualization and dimensionality reduction
- Enhance preprocessing using techniques like **CLAHE** (contrast enhancement)

---

##  Acknowledgements

- **Dataset**: Provided by Tawsifur Rahman on [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **Libraries Used**:
  - `OpenCV`
  - `scikit-image`
  - `scikit-learn`
  - `xgboost`
  - `matplotlib`, `seaborn` for visualization

---

##  Final Note

This project showcases that with the right **feature engineering**, even traditional machine learning pipelines can achieve high performance in **image classification tasks**, particularly in the **medical domain**. It is ideal for:
- **Edge devices**
- **Low-resource deployments**
- **Educational purposes** to understand the fundamentals of computer vision and ML integration.

---


        
