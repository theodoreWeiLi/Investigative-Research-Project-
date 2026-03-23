## Plan: VOICED Dataset Machine Learning Mini-Project

This project applies supervised and unsupervised ML techniques to short voice recordings (VOICED dataset) to detect and cluster vocal fold pathologies. Using an Apple M4 Pro (48GB RAM), the dataset of 208 samples will process and train extremely fast, allowing us to focus on step-by-step learning, data exploration, and high-quality feature extraction. 

### Phase 1: Environment Setup & Data Loading
1. Set up a Python virtual environment (e.g., using `conda` or standard `venv`) and install core libraries (`wfdb`, `librosa`, `scikit-learn`, `pandas`, `matplotlib`, `jupyter`). We will ensure M4/ARM64 compatibility.
2. Initialize a Jupyter Notebook to allow interactive step-by-step learning.
3. Write a script to iterate through the `voice-icar-federico-ii-database-1.0.0` folder and load the `*-info.txt` (metadata) and `.dat` (audio signal) files using the `wfdb` library.

### Phase 2: Data Exploration & Preprocessing
4. Explore the metadata: Visualize the distribution of healthy vs. pathological cases, age, gender, and smoking habits.
5. Signal processing: Extract acoustic features from the raw 5-sec audio files (e.g., Mel-Frequency Cepstral Coefficients (MFCCs), fundamental frequency, jitter, shimmer) using `librosa`.
6. Data compilation: Build a clean Pandas DataFrame merging the extracted features with the target labels (pathology status).
7. Preprocessing: Handle any missing values and normalize the feature sets (e.g., using `StandardScaler`).

### Phase 3: Supervised Learning (Problem #1)
8. Split the dataset into training and testing sets (e.g., 80/20 split).
9. Train 5 different classification models (e.g., Logistic Regression, SVM, Random Forest, K-Nearest Neighbors, and a Multi-Layer Perceptron/MLP). 
10. Evaluate models using appropriate metrics (Accuracy, Sensitivity, Specificity, F1-Score) and generate confusion matrices.

### Phase 4: Unsupervised Learning (Problem #2)
11. Apply clustering algorithms (e.g., K-Means, DBSCAN, or hierarchical clustering) on the features without providing the labels.
12. Reduce dimensionality (e.g., PCA or t-SNE) to visualize the clusters in 2D space.
13. Analyze the clusters: Cross-reference the discovered clusters with the metadata (e.g., do clusters map to smokers, specific ages, or actual disease?).

### Phase 5: Report Generation
14. Structure a 6-page IEEE JBHI format report draft.
15. Populate the report with methodology details, tables of the 5-model comparison, and visualizations from clustering.
16. Compare the findings with the provided literature (e.g., the 2018 benchmark paper by Cesari et al.).
