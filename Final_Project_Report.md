# Machine Learning Analysis of Physiological Voice Data for Pathological Detection

**Authors:** [Group Members' Names]  
**Course:** PHM5015 Investigative Research Project

## Abstract
This report presents a comprehensive machine learning analysis of the VOICED (Voice ICar Federico II) database to distinguish between healthy and pathological voices. Acoustic features including Mel-Frequency Cepstral Coefficients (MFCCs), Spectral Centroid, Spectral Bandwidth, and Zero Crossing Rate (ZCR) were extracted from raw physiological audio signals. Due to extreme class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) and algorithm-level class weighting were employed. Five supervised learning models (Logistic Regression, Support Vector Machine, Random Forest, K-Nearest Neighbors, and Multi-Layer Perceptron) and one unsupervised model (K-Means) were evaluated. The application of SMOTE successfully improved model specificity, mitigating the algorithmic bias toward the majority pathological class. Support Vector Machine demonstrated the highest sensitivity (0.83), while K-Nearest Neighbors yielded the highest specificity (0.75). These findings highlight the potential and challenges of using voice as a non-invasive digital biomarker for precision medicine.

## 1. Introduction
Voice and speech analysis are increasingly gaining attention as non-invasive biomarkers for health monitoring. With the proliferation of high-quality microphones in personal devices, voice-based diagnostics offer a scalable and accessible approach for early disease detection. Vocal fold pathologies and dysphonias can be non-invasively screened using physiological audio recordings, providing clinical decision support systems with objective diagnostic metrics.

This study utilizes the Voice ICar fEDerico II (VOICED) database, which comprises 208 voice recordings from healthy individuals and patients with clinically diagnosed vocal fold pathologies. The dataset presents significant class imbalance, predominantly containing pathological instances. The primary objective is to implement a robust machine learning pipeline utilizing both supervised and unsupervised techniques to classify voices as healthy or pathological, specifically addressing the challenges of skewed clinical data.

## 2. Methods

### 2.1 Dataset Description
The VOICED database consists of 208 recordings, each featuring a 5-second sustained vocalization of the vowel /a/, recorded in a controlled acoustic environment. Extensive metadata is provided, including age, gender, clinical diagnosis, smoker status, and subjective questionnaire scores (VHI and RSI). The dataset exhibits a severe class imbalance, with 57 healthy cases and 151 pathological cases.

### 2.2 Feature Extraction
Raw audio waveforms were processed using the `librosa` library in Python. To represent the audio mathematically, a feature array of 16 variables was extracted for each patient:
*   **13 Mel-Frequency Cepstral Coefficients (MFCCs):** Averaged over time to capture the short-term power spectrum of the sound.
*   **Mean Spectral Centroid:** Indicating the "center of mass" of the frequency spectrum.
*   **Mean Spectral Bandwidth:** Measuring the width of the frequency band.
*   **Mean Zero Crossing Rate (ZCR):** Capturing the rate at which the signal changes sign.

### 2.3 Data Preprocessing & Balancing
Features were normalized using a standard normal distribution (`StandardScaler`) to prevent variables with larger magnitudes from dominating the learning process. The dataset was partitioned into an 80% training set and a 20% test set utilizing stratified splitting to preserve the original diagnostic ratio.

To address the class imbalance, two strategies were combined:
1.  **SMOTE (Synthetic Minority Over-sampling Technique):** Applied exclusively to the training set, it synthetically expanded the minority (healthy) class to create a strictly balanced training environment.
2.  **Class Weights:** `class_weight='balanced'` was applied to applicable mathematical models (Logistic Regression, SVM, Random Forest) to computationally penalize minority class misclassification.

### 2.4 Model Selection
**Supervised Learning:** Five distinct classification algorithms were trained to address the binary classification problem (Healthy vs. Pathological):
*   Logistic Regression (LR)
*   Support Vector Machine (SVM with RBF kernel)
*   Random Forest (RF, 100 estimators)
*   K-Nearest Neighbors (KNN, k=5)
*   Multi-Layer Perceptron (MLP, 100x50 hidden layers)

**Unsupervised Learning:** K-Means clustering ($k=2$) was performed on the fully scaled dataset to detect natural groupings without providing true diagnosis labels. Principal Component Analysis (PCA) was subsequently used to compress the 16 features into a 2D space for visual interpretation and cross-tabulation against clinical ground truth.

## 3. Results

### 3.1 Supervised Learning Performance
The models were evaluated on the 20% hold-out test set using Accuracy, Sensitivity (Recall for pathological), Specificity (Recall for healthy), and F1-Score. The implementation of SMOTE and class weights significantly elevated the specificity across models, marking a dramatic improvement over baseline models which strictly favored the pathological majority class.

| Algorithm | Accuracy | Sensitivity | Specificity | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.67 | 0.70 | 0.58 | 0.75 |
| **Support Vector Machine** | 0.69 | 0.83 | 0.33 | 0.79 |
| **Random Forest** | 0.62 | 0.70 | 0.42 | 0.72 |
| **K-Nearest Neighbors** | 0.60 | 0.53 | 0.75 | 0.65 |
| **Neural Network (MLP)** | 0.67 | 0.80 | 0.33 | 0.77 |

Support Vector Machine achieved the highest sensitivity (0.83) and F1-Score (0.79), demonstrating robustness in identifying true pathological cases. Conversely, K-Nearest Neighbors achieved the highest specificity (0.75), making it the most conservative model for confirming healthy patients, though at the cost of overall accuracy and sensitivity.

### 3.2 Unsupervised Clustering
The K-Means clustering algorithm partitioned the data into two mathematical centroids. When visualized via PCA, significant overlap between the clusters was observed, implying that the pure acoustic difference between mild dysphonias and healthy voices is not linearly separable without supervised labeled boundaries. Cross-tabulation revealed that unsupervised clustering alone struggles to perfectly mirror clinical diagnoses due to these overlapping complex features.

## 4. Discussion and Interpretation
Voice analysis presents a promising frontier in precision medicine, offering non-intrusive diagnostic capabilities. However, addressing class imbalance is structurally critical when evaluating physiological datasets like VOICED. Prior to balancing, models theoretically achieved near 100% sensitivity but <20% specificity by taking the "mathematically safe" route of predicting the majority class. With SMOTE, a clinical trade-off was achieved.

The contrasting performance of KNN (high specificity) and SVM (high sensitivity) highlights the need for ensemble methods or adjustable decision thresholds in clinical applications. In a screening context, high sensitivity is often prioritized to avoid missing pathological cases, favoring models like SVM.

The PCA and clustering results further prove the necessity of supervised learning; unsupervised clustering struggles with the overlapping acoustic markers characteristic of mild dysphonias. 

**Limitations and Future Work:** The current feature set is limited to standard acoustic properties. Future iterations should explore extracting advanced perturbation features (jitter, shimmer) and employing deep learning techniques directly on spectrograms to improve separability. Additionally, testing on a larger, more balanced dataset would enhance model generalizability.

## 5. Conclusion
This study successfully developed a machine learning pipeline to classify vocal fold pathologies using the VOICED dataset. By combining robust feature extraction with class balancing techniques (SMOTE), the models demonstrated viable diagnostic performance. The SVM model emerged as the most sensitive classifier, underscoring the potential of voice as a digital biomarker in precision medicine. Further research incorporating advanced feature sets and larger cohorts is recommended to realize clinical deployment.

## 6. References
[1] A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals,” Circulation, vol. 101, no. 23, pp. e215–e220, Jun. 2000.  
[2] L. Verde and G. Sannino, “VOICED Database.” PhysioNet. doi: 10.13026/TWFD-KB89.  
[3] U. Cesari, G. De Pietro, E. Marciano, C. Niri, G. Sannino, and L. Verde, “A new database of healthy and pathological voices,” Computers & Electrical Engineering, vol. 68, pp. 310–321, May 2018, doi: 10.1016/j.compeleceng.2018.04.008.  
[4] B. McFee et al., "librosa: Audio and Music Signal Analysis in Python," in Proceedings of the 14th Python in Science Conference, 2015.  
[5] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.  
[6] N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," Journal of Artificial Intelligence Research, vol. 16, pp. 321-357, 2002.

---
**Author Contributions:**  
[Student 1] contributed to the data preprocessing and feature extraction. [Student 2] implemented the supervised and unsupervised machine learning models. [Student 3] conducted the evaluation and drafted this report. All authors reviewed and approved the final manuscript.