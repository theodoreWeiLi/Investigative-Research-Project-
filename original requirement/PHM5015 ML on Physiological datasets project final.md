Group Project – Machine Learning on Physiological Datasets [Due: 10th April, Friday, 2 pm]

Maximum Score: 100 points [40% of overall grade]

Group Submission

This project aims to provide you with hands-on experience in applying machine learning (ML) techniques to physiological datasets, with a focus on exploring and understanding digital biomarkers for precision medicine. Through this, you'll gain practical skills in data handling, feature extraction, model development, and interpretation of results within the context of digital health.

In groups of 3, students will work with a selected physiological dataset (e.g., heart rate, electrodermal activity, EEG, PPG, or multimodal wearable sensor data). You will design, train, and evaluate ML models using a dataset selected from a curated list of PhysioNet databases [1]. Each dataset includes one or more biosignals along with metadata, allowing for a range of precision medicine applications-from disease classification to the estimation of physiological age or risk profiles.

Your task is to:

1. Understand the dataset: Explore data structure, variables, and potential healthrelated patterns.

2. Define a problem: Choose a relevant ML task such as classification, regression, or clustering.

3. Preprocess and analyze: Handle missing data, normalize, and extract meaningful features.

4. Build and evaluate ML models: Use appropriate algorithms (e.g., SVM, Random Forest, Neural Networks) and validate their performance.

5. Interpret and report: Relate findings back to physiological or clinical relevance- what could these biomarkers tell us in a real-world setting?

Page 1 of 6

## Deliverables:

• A working Jupyter Notebook or script with code and explanations

o Your source code should be in Python, and you are free to use any opensource libraries. However, the source code must be written by your team and should not copy published solutions. Please ensure that your source code is well organized and commented! Include a clear README file with instruction on how to reproduce your results as well as an overview of all dependencies and software versions used.

• A report (max 6 pages, excluding references) covering methodology, results, and interpretation

o Each group must submit a concise report summarizing your methodology, results, and interpretation of findings. The report should be written in the format of the IEEE Journal of Biomedical and Health Informatics (JBHI). This format is commonly used in biomedical engineering and digital health research, and following it will give you experience in scientific writing aligned with real-world standards.

Your report should include:

1. Title and Abstract – A clear and informative title, and a brief abstract summarizing your study.

2. Introduction – Context, motivation, and problem statement.

3. Methods – Description of the dataset, preprocessing steps, feature engineering, and machine learning models used.

4. Results – Performance metrics, model comparison, and key findings (with tables and figures as needed).

5. Discussion and Interpretation – What do your results mean in the context of precision medicine? What are the limitations and potential applications?

6. Conclusion – Summary of contributions and possible future directions.

7. References – Cite relevant studies, tools, and datasets using IEEE citation style.

Contributions of each team member should be briefly noted at the end of the report.

Good luck!

Page 2 of 6

## The project will be assessed based on the following criteria:

### Algorithm Development Performance: [50 points]

• Approach and Justification (15 pts): Clear explanation of model choice, preprocessing steps, feature selection, and training methods.

• Code Quality (10 pts): Clarity, readability, and absence of unnecessary complexity or redundancy in your source code, and whether the code runs correctly without bugs or errors.

• Model Performance (15 pts): Quality and appropriateness of selected evaluation metrics, such as confusion matrix (high accuracy, sensitivity, specificity, F1 score, precision, recall) for classification tasks; RMSE for regression tasks; and Adjusted Rand Index (ARI) or Silhouette Score for clustering tasks.

• Innovative Techniques (10 pts): Use of creative strategies like ensemble methods, optimization, normalization, handling imbalanced datasets, etc.

### Analysis and Comparison: [20 points]

• Critical Comparison (10 pts): Meaningful benchmarking of the model’s performance with published models in literature.

• Interpretation of Results (10 pts): Logical discussion of strengths, weaknesses, possible reasons for performance differences.

### Report: [30 points]

• Organized, well-written, and stays within 5-page limit. Includes abstract, methods, results, discussion, and references.

• Demonstrates deep understanding, not just surface-level description.

• Proper referencing of datasets, algorithms, and any external work.

Note: For any clarifications, feel free to reach out to the instructors for guidance.

Page 3 of 6

## Physionet Datasets

### Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging

The Autonomic Aging dataset was created to support research into how healthy aging affects cardiovascular function. With age, the autonomic regulation of blood pressure and cardiac rhythm gradually declines-a process associated with increased risk of age-related disorders such as dementia and Alzheimer’s disease. The dataset includes high-resolution electrocardiogram (ECG) and continuous non-invasive blood pressure recordings from 1121 healthy volunteers at rest, across a broad age range. The data were collected by researchers from the department of psychosomatic medicine and psychotherapy at Friedrich Schiller University Jena, Germany.

The dataset, including metadata, and details regarding the data collection protocols and signal formats can be found on the PhysioNet project page [2] and in the accompanying publication [3].

#### Problem # 1 – Supervised learning

Use supervised learning techniques to predict age group based on ECG-derived features - such as heart rate variability (HRV)-and/or the raw ECG signals. Investigate and compare the performance of 5 different models.

#### Problem # 2 – Unsupervised learning

Apply unsupervised machine learning techniques to identify patterns or clusters within ECGderived features-such as heart rate variability (HRV)-or the raw ECG signals. Your objective is to explore whether distinct age profiles emerge from the data, and how these clusters may relate to participant age group, sex, or BMI (these metadata are provided in the database). Discuss the clinical relevance of the clusters you discover.

Page 4 of 6

### A new database of healthy and Pathological voices

The Voice ICar fEDerico II (VOICED) database comprises 208 voice recordings from both healthy individuals and patients with clinically diagnosed vocal fold pathologies. Voice is gaining attention as a non-invasive biomarker for health monitoring, particularly due to its sensitivity to a wide range of physiological and neurological conditions. With the widespread availability of high-quality microphones in smartphones and other personal devices, voicebased diagnostics are becoming increasingly accessible and scalable.

To advance research in voice pathology detection, the VOICED database was developed using a standardized protocol. Each participant provided a 5-second sustained vocalization of the vowel /a/, recorded in a controlled acoustic environment. The dataset also includes metadata, such as age, gender, clinical diagnosis, lifestyle factors (e.g., smoking and alcohol use), occupation, and responses to the Voice Handicap Index (VHI) and Reflux Symptom Index (RSI) questionnaires.

The dataset, including metadata, and details regarding the data collection protocols and signal formats can be found on the PhysioNet project page [4] and in the accompanying publication [5].

#### Problem # 1 – Supervised learning

Use supervised learning techniques to classify pathology based on voice-derived features and/or the raw voice signals. Investigate and compare the performance of 5 different models.

#### Problem # 2 – Unsupervised learning

Apply unsupervised machine learning techniques to identify patterns or clusters within voicederived features or the raw signals. Your objective is to explore whether distinct pathology emerges from the data, and how these clusters may relate to the metadata provided in the database. Discuss the clinical relevance of the clusters you discover.

Page 5 of 6

## References

[1] A. L. Goldberger et al., “PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals,” Circulation, vol. 101, no. 23, pp. e215–e220, Jun. 2000.

[2] A. Schumann and K. Bär, “Autonomic Aging: A dataset to quantify changes of cardiovascular autonomic function during healthy aging.” PhysioNet. doi: 10.13026/2HSY-T491.

[3] A. Schumann and K.-J. Bär, “Autonomic aging – A dataset to quantify changes of cardiovascular autonomic function during healthy aging,” Sci Data, vol. 9, no. 1, p. 95, Mar. 2022, doi: 10.1038/s41597-022-01202-y.

[4] L. Verde and G. Sannino, “VOICED Database.” PhysioNet. doi: 10.13026/TWFD-KB89.

[5] U. Cesari, G. De Pietro, E. Marciano, C. Niri, G. Sannino, and L. Verde, “A new database of healthy and pathological voices,” Computers & Electrical Engineering, vol. 68, pp. 310–321, May 2018, doi: 10.1016/j.compeleceng.2018.04.008.
