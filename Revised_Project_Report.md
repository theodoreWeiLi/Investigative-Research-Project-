# Machine Learning Analysis of Physiological Voice Data for Pathological Detection

## Abstract
This report presents a comprehensive machine learning analysis of the VOICED (Voice ICar Federico II Database) dataset to distinguish between healthy and pathological voices. Acoustic features including Mel-Frequency Cepstral Coefficients (MFCCs), Spectral Centroid, Spectral Bandwidth, and Zero Crossing Rate (ZCR) were extracted from raw physiological audio signals. Due to extreme class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) and algorithm-level class weighting were employed. Five supervised learning models (Logistic Regression, SVM, Random Forest, KNN, MLP) and one unsupervised model (K-Means) were evaluated. The application of SMOTE successfully improved model specificity, mitigating the algorithmic bias toward the majority pathological class. Support Vector Machine demonstrated the highest sensitivity (0.83), while K-Nearest Neighbors yielded the highest specificity (0.75).

## 1. Introduction
Vocal fold pathologies and dysphonias can be non-invasively screened using physiological audio recordings. Implementing machine learning algorithms on such datasets provides clinical decision support systems with objective diagnostic metrics. The VOICED dataset contains 208 patient records with significant class imbalance (predominantly pathological instances). The goal of this mini-project is to implement a robust pipeline utilizing both supervised and unsupervised learning techniques to classify voices as healthy or pathological, specifically addressing the challenges of skewed clinical data.

## 2. Methods

### 2.1. Feature Extraction
Raw audio waveforms were processed using the `librosa` library. Instead of processing raw time-series data, mathematical representations of the sound were extracted. The finalized feature array for each patient consisted of 16 variables:
*   13 Mel-Frequency Cepstral Coefficients (MFCCs) averaged over time.
*   Mean Spectral Centroid (indicating the "center of mass" of the spectrum).
*   Mean Spectral Bandwidth.
*   Mean Zero Crossing Rate (ZCR).

### 2.2. Data Preprocessing & Balancing
Features were scaled using standard normal distribution (StandardScaler) to prevent magnitude dominance among different variables. The dataset was partitioned into an 80% training set and a 20% test set using stratified splitting to preserve the original diagnostic ratio. 

To address the severe class imbalance (57 healthy vs. 151 pathological), two strategies were combined:
1.  **SMOTE (Synthetic Minority Over-sampling Technique):** Applied exclusively to the training set, synthetically expanding the minority (healthy) class to create a strictly balanced training environment (242 total training samples).
2.  **Class Weights:** The `class_weight='balanced'` parameter was applied to applicable mathematical models (Logistic Regression, SVM, Random Forest) to computationally penalize minority class misclassification.

### 2.3. Model Selection
**Supervised Learning (Problem 1):** Five distinct classification algorithms were trained: Logistic Regression (LR), Support Vector Machine (SVM with RBF kernel), Random Forest (RF, 100 estimators), K-Nearest Neighbors (KNN, k=5), and Multi-Layer Perceptron (MLP, 100x50 hidden layers).

**Unsupervised Learning (Problem 2):** K-Means clustering ($k=2$) was performed on the fully scaled dataset to detect natural algorithmic groupings without providing true diagnosis labels. Principal Component Analysis (PCA) was used to compress the 16 features into a 2D space for visual cross-tabulation against clinical ground truth.

## 3. Results

### 3.1. Supervised Learning Performance
The models were evaluated using Accuracy, Sensitivity (Recall for pathological), Specificity (Accuracy for healthy), and F1-Score. The implementation of SMOTE successfully elevated the specificity across models, marking a dramatic improvement over baseline models which strictly favored the pathological class.

| Algorithm | Accuracy | Sensitivity | Specificity | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.67 | 0.70 | 0.58 | 0.75 |
| **Support Vector Machine** | 0.69 | 0.83 | 0.33 | 0.79 |
| **Random Forest** | 0.62 | 0.70 | 0.42 | 0.72 |
| **K-Nearest Neighbors** | 0.60 | 0.53 | 0.75 | 0.65 |
| **Neural Network (MLP)** | 0.67 | 0.80 | 0.33 | 0.77 |

### 3.2. Unsupervised Clustering
The K-means clustering defined two mathematical centroids. When visualized via PCA, there was notable overlap between the clusters, implying that the pure acoustic difference between mild dysphonias and healthy voices is not linearly separable without supervised labeled boundaries. The cross-tabulation showed that unsupervised clustering alone struggles to perfectly mirror the clinical diagnoses due to these overlapping complex features.

## 4. Discussion and Conclusion
The inclusion of SMOTE and class balancing is structurally critical when evaluating physiological datasets like VOICED. Prior to balancing, models achieved near 100% sensitivity but <20% specificity by taking the "mathematically safe" route of predicting the majority class. With SMOTE, a clinical trade-off was achieved. 

The KNN algorithm demonstrated the highest success in properly identifying healthy patients (0.75 Specificity), while the SVM provided the greatest robust metric for correctly labeling disease presence (0.83 Sensitivity / 0.69 Accuracy). The PCA results further prove the necessity of supervised learning, as unsupervised clustering struggles with overlapping acoustic markers. Future iterations should explore deeper hyperparameter tuning and the extraction of jitter and shimmer voice features to improve separability.

## 5. References
1. McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. Proceedings of the 14th Python in Science Conference.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
3. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.
4. Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation.