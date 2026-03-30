# Machine Learning Analysis of Physiological Voice Data for Pathological Detection

## Abstract
This report presents an enhanced machine learning analysis of the VOICED (Voice ICar Federico II Database) dataset to classify healthy versus pathological voices. Acoustic features were fused with metadata (age, gender, smoker status, VHI, RSI), and a stratified train/validation/test framework with 5-fold cross-validated hyperparameter tuning was used. Eight supervised models were evaluated (Logistic Regression, SVM-RBF, SVM-Cubic, Random Forest, KNN, MLP, XGBoost, CatBoost) with SMOTE and threshold tuning to improve specificity while preserving sensitivity. KNN and SVM-Cubic achieved the highest specificity (0.667), while KNN delivered the strongest balanced profile (Accuracy 0.738, Sensitivity 0.767, Specificity 0.667, F1 0.807).

## 1. Introduction
Vocal fold pathologies and dysphonias can be non-invasively screened using physiological audio recordings. Implementing machine learning algorithms on such datasets provides clinical decision support systems with objective diagnostic metrics. The VOICED dataset contains 208 patient records with significant class imbalance (predominantly pathological instances). The goal of this mini-project is to implement a robust pipeline utilizing both supervised and unsupervised learning techniques to classify voices as healthy or pathological, specifically addressing the challenges of skewed clinical data.

## 2. Methods

### 2.1. Feature Extraction
Raw audio waveforms were processed using `librosa`. Instead of raw time-series input, engineered acoustic features were extracted: MFCC means and standard deviations (13 coefficients), spectral centroid mean/std, spectral bandwidth mean/std, and zero crossing rate mean/std. Clinical metadata (age, gender, smoker, VHI, RSI) were integrated into the supervised feature set.

### 2.2. Data Preprocessing & Balancing
Preprocessing used a leakage-safe pipeline: median imputation + standard scaling for numeric variables and mode imputation + one-hot encoding for categorical variables (gender/smoker). Data were split into stratified train/validation/test subsets (60/20/20). 

Class imbalance (57 healthy vs. 151 pathological) was handled using:
1.  **SMOTE:** Applied only to training folds.
2.  **Class Weights:** Used where supported by model formulations.
3.  **Decision Threshold Tuning:** Validation-based threshold selection optimized specificity under sensitivity $\geq 0.70$.

### 2.3. Model Selection
**Supervised Learning (Problem 1):** Eight classification algorithms were trained: Logistic Regression, SVM (RBF), SVM (cubic polynomial, degree = 3), Random Forest, KNN, MLP, XGBoost, and CatBoost. Hyperparameters were tuned with 5-fold cross-validation using balanced accuracy. Key tuned parameters included Random Forest `max_depth`, SVM `C`/`gamma`, cubic SVM `coef0`, KNN neighbor count, and boosting model depth/learning rate/iterations.

**Unsupervised Learning (Problem 2):** K-Means clustering ($k=2$) was performed on the fully scaled dataset to detect natural algorithmic groupings without providing true diagnosis labels. Principal Component Analysis (PCA) was used to compress the 16 features into a 2D space for visual cross-tabulation against clinical ground truth.

## 3. Results

### 3.1. Supervised Learning Performance
The models were evaluated on the hold-out test set using Accuracy, Sensitivity, Specificity, and F1-score after validation-based threshold tuning.

| Algorithm | Accuracy | Sensitivity | Specificity | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.643 | 0.800 | 0.250 | 0.762 |
| **Support Vector Machine (RBF)** | 0.595 | 0.633 | 0.500 | 0.691 |
| **Support Vector Machine (Cubic)** | 0.714 | 0.733 | 0.667 | 0.786 |
| **Random Forest** | 0.714 | 0.767 | 0.583 | 0.793 |
| **K-Nearest Neighbors** | 0.738 | 0.767 | 0.667 | 0.807 |
| **Neural Network (MLP)** | 0.619 | 0.733 | 0.333 | 0.733 |
| **XGBoost** | 0.762 | 0.900 | 0.417 | 0.844 |
| **CatBoost** | 0.714 | 0.833 | 0.417 | 0.806 |

### 3.2. Unsupervised Clustering
The K-means clustering defined two mathematical centroids. When visualized via PCA, there was notable overlap between the clusters, implying that the pure acoustic difference between mild dysphonias and healthy voices is not linearly separable without supervised labeled boundaries. The cross-tabulation showed that unsupervised clustering alone struggles to perfectly mirror the clinical diagnoses due to these overlapping complex features.

## 4. Discussion and Conclusion
The inclusion of SMOTE and class balancing is structurally critical when evaluating physiological datasets like VOICED. Prior to balancing, models achieved near 100% sensitivity but <20% specificity by taking the "mathematically safe" route of predicting the majority class. With SMOTE, a clinical trade-off was achieved. 

The enhanced pipeline indicates that metadata fusion and tuned hyperparameters provide measurable gains. The cubic SVM variant improved substantially over RBF-SVM in specificity and accuracy, aligning with recent VOICED-oriented literature that favors more flexible decision boundaries after richer feature engineering. KNN and SVM-Cubic were the best choices when prioritizing healthy-class recognition, while XGBoost remained useful when maximizing pathological-case sensitivity. The PCA results still suggest that unsupervised clustering alone has limited diagnostic alignment due to feature overlap. Future work should include jitter/shimmer features, repeated external validation, and larger cohorts.

## 5. References
1. McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. Proceedings of the 14th Python in Science Conference.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
3. Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. Journal of Artificial Intelligence Research, 16, 321-357.
4. Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation.
5. Chen, T., and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
6. Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS.