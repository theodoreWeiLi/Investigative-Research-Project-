# Machine Learning Analysis of Physiological Voice Data for Pathological Detection

**Authors:** [Group Members' Names]  
**Course:** PHM5015 Investigative Research Project

## Abstract
This report presents an enhanced machine learning analysis of the VOICED (Voice ICar Federico II) database to distinguish healthy and pathological voices. In addition to acoustic features (MFCCs, spectral centroid, spectral bandwidth, and zero crossing rate), clinical metadata (age, gender, smoker status, VHI, RSI) were incorporated. A stratified train/validation/test protocol with cross-validated hyperparameter tuning was implemented, including explicit optimization of tree depth (max_depth) and decision thresholds to improve specificity while maintaining acceptable sensitivity. Eight supervised models were evaluated: Logistic Regression, Support Vector Machine (RBF), Support Vector Machine (Cubic polynomial), Random Forest, K-Nearest Neighbors, Multi-Layer Perceptron, XGBoost, and CatBoost. K-Nearest Neighbors and SVM-Cubic achieved the highest specificity (0.667), while KNN provided the strongest specificity-sensitivity balance (Accuracy 0.738, Sensitivity 0.767, Specificity 0.667, F1 0.807). These findings demonstrate that metadata fusion and model-level tuning can materially improve robustness for voice-based pathological screening.

## 1. Introduction
Voice and speech analysis are increasingly gaining attention as non-invasive biomarkers for health monitoring. With the proliferation of high-quality microphones in personal devices, voice-based diagnostics offer a scalable and accessible approach for early disease detection. Vocal fold pathologies and dysphonias can be non-invasively screened using physiological audio recordings, providing clinical decision support systems with objective diagnostic metrics.

This study utilizes the Voice ICar fEDerico II (VOICED) database, which comprises 208 voice recordings from healthy individuals and patients with clinically diagnosed vocal fold pathologies. The dataset presents significant class imbalance, predominantly containing pathological instances. The primary objective is to implement a robust machine learning pipeline utilizing both supervised and unsupervised techniques to classify voices as healthy or pathological, specifically addressing the challenges of skewed clinical data.

## 2. Methods

### 2.1 Dataset Description
The VOICED database consists of 208 recordings, each featuring a 5-second sustained vocalization of the vowel /a/, recorded in a controlled acoustic environment. Extensive metadata is provided, including age, gender, clinical diagnosis, smoker status, and subjective questionnaire scores (VHI and RSI). The dataset exhibits a severe class imbalance, with 57 healthy cases and 151 pathological cases.

### 2.2 Feature Extraction
Raw audio waveforms were processed using the `librosa` library in Python. To improve representation quality, both central tendency and variability were extracted. The final feature space included:
*   **MFCC features:** 13 coefficients with mean and standard deviation per coefficient.
*   **Spectral features:** mean and standard deviation of spectral centroid and spectral bandwidth.
*   **Temporal feature:** mean and standard deviation of zero crossing rate (ZCR).
*   **Metadata features:** age, gender, smoker status, Voice Handicap Index (VHI), and Reflux Symptom Index (RSI).

### 2.3 Data Preprocessing & Balancing
Preprocessing was implemented through a leakage-safe pipeline. Numeric features were median-imputed and standardized; categorical features (gender, smoker) were mode-imputed and one-hot encoded. Data were split into stratified train/validation/test subsets (60/20/20).

Class imbalance (57 healthy vs. 151 pathological) was addressed by:
1.  **SMOTE:** applied only on training folds within the modeling pipeline.
2.  **Class weighting:** applied where model formulations support it.
3.  **Threshold optimization:** decision thresholds were tuned on the validation set to maximize specificity under a sensitivity constraint ($\geq 0.70$).

### 2.4 Model Selection
**Supervised Learning:** Eight classification algorithms were trained for binary diagnosis (healthy vs. pathological):
*   Logistic Regression (LR)
*   Support Vector Machine (SVM, RBF kernel)
*   Support Vector Machine (SVM, cubic polynomial kernel, degree = 3)
*   Random Forest (RF)
*   K-Nearest Neighbors (KNN)
*   Multi-Layer Perceptron (MLP)
*   XGBoost
*   CatBoost

Hyperparameters were tuned via 5-fold cross-validation (balanced accuracy objective). Important tuned parameters included RF `max_depth`, SVM `C`/`gamma`, KNN neighbors, and boosting depth/learning rate/iterations.

**Unsupervised Learning:** K-Means clustering ($k=2$) was performed on the fully scaled dataset to detect natural groupings without providing true diagnosis labels. Principal Component Analysis (PCA) was subsequently used to compress the 16 features into a 2D space for visual interpretation and cross-tabulation against clinical ground truth.

## 3. Results

### 3.1 Supervised Learning Performance
The models were evaluated on the 20% hold-out test set using Accuracy, Sensitivity (recall for pathological), Specificity (recall for healthy), and F1-score after validation-based threshold tuning.

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

KNN and SVM-Cubic jointly achieved the highest specificity (0.667), with KNN showing the best overall balance in this run (accuracy 0.738, F1 0.807). The cubic kernel variant improved substantially over the baseline RBF-SVM for both specificity and accuracy. XGBoost remained the most sensitivity-oriented model (0.900) but with lower specificity (0.417).

### 3.2 Unsupervised Clustering
The K-Means clustering algorithm partitioned the data into two mathematical centroids. When visualized via PCA, significant overlap between the clusters was observed, implying that the pure acoustic difference between mild dysphonias and healthy voices is not linearly separable without supervised labeled boundaries. Cross-tabulation revealed that unsupervised clustering alone struggles to perfectly mirror clinical diagnoses due to these overlapping complex features.

## 4. Discussion and Interpretation
Voice analysis presents a promising frontier in precision medicine, offering non-intrusive diagnostic capabilities. However, addressing class imbalance is structurally critical when evaluating physiological datasets like VOICED. Prior to balancing, models theoretically achieved near 100% sensitivity but <20% specificity by taking the "mathematically safe" route of predicting the majority class. With SMOTE, a clinical trade-off was achieved.

The revised pipeline shows that combining metadata (including smoker status), explicit hyperparameter tuning (including RF `max_depth`), and threshold adjustment can improve clinical balance. Introducing a cubic-kernel SVM, inspired by recent VOICED-focused literature, improved boundary flexibility and yielded better specificity than RBF-SVM in this dataset.

The performance contrast across models still reflects a clinical trade-off: models optimized for sensitivity (XGBoost) may increase false positives, while models with stronger specificity (KNN and SVM-Cubic) are more conservative in labeling healthy cases. For screening contexts, a high-sensitivity operating point may remain preferred; for confirmatory triage, specificity-oriented operating points are clinically valuable.

The PCA and clustering results further prove the necessity of supervised learning; unsupervised clustering struggles with the overlapping acoustic markers characteristic of mild dysphonias. 

**Limitations and Future Work:** The cohort size remains limited (208 recordings), which constrains model generalizability and hyperparameter stability. Future work should add perturbation features such as jitter and shimmer, evaluate repeated or nested cross-validation, and validate on external cohorts. Deep learning on spectrograms can be explored after stronger data scaling.

## 5. Conclusion
This study developed and revised a machine learning pipeline for pathological voice classification in VOICED. By integrating acoustic and metadata features, applying balanced preprocessing with SMOTE, and tuning both hyperparameters and decision thresholds, the enhanced models improved the sensitivity-specificity trade-off. KNN and SVM-Cubic provided the strongest specificity in the final evaluation, while XGBoost provided the highest sensitivity. These results support the potential of voice as a practical digital biomarker while highlighting the need for larger datasets and richer voice perturbation features for clinical deployment.

## 6. References
[1] A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals,” Circulation, vol. 101, no. 23, pp. e215–e220, Jun. 2000.  
[2] L. Verde and G. Sannino, “VOICED Database.” PhysioNet. doi: 10.13026/TWFD-KB89.  
[3] U. Cesari, G. De Pietro, E. Marciano, C. Niri, G. Sannino, and L. Verde, “A new database of healthy and pathological voices,” Computers & Electrical Engineering, vol. 68, pp. 310–321, May 2018, doi: 10.1016/j.compeleceng.2018.04.008.  
[4] B. McFee et al., "librosa: Audio and Music Signal Analysis in Python," in Proceedings of the 14th Python in Science Conference, 2015.  
[5] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.  
[6] N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," Journal of Artificial Intelligence Research, vol. 16, pp. 321-357, 2002.
[7] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," Proc. KDD, pp. 785-794, 2016.
[8] L. Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features," NeurIPS, vol. 31, 2018.

---
**Author Contributions:**  
[Student 1] contributed to the data preprocessing and feature extraction. [Student 2] implemented the supervised and unsupervised machine learning models. [Student 3] conducted the evaluation and drafted this report. All authors reviewed and approved the final manuscript.