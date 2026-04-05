# Rainfall Prediction in Australia using Deep Learning
This repository contains the implementation of a deep learning project/assignment aimed at predicting rainfall in Australia using historical weather data. This project was developed as part of an Individual assignment for the Optimization and Deep Learning module. 

## Project Overview
Accurate rainfall prediction is critical for sectors like agriculture and water management in Australia. This research explores the effectiveness of Recurrent Neural Networks (RNNs)-specifically Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)-in capturing temporal dependencies within weather data to predict whether it will rain the following day ("RainTomorrow")

### Objectives
- **Data Preparation**: Clean, normalize, and transform raw weather data into high-quality inputs for deep learning models.
- **Exploratory Data Analysis (EDA)**: Identify correlations between attributes such as humidity, pressure, and wind patterns.
- **Model Comparison**: Implement and compare the performance of LSTM and GRU architectures.
- **Model Optimization**: Use Keras Tuner to automate hyperparameter optimization
- **Evaluation**: Assess performance using Accuracy, Precision, Recall, and F1-score

# Dataset
- **Source**: Australian Bureau of Meteorology (via Kaggle)
- **Dimensions**: Approximately 145,160 observations, 23 features, including temperature, humidity, pressure, and wind patterns.
- **Target Variable**: RainTomorrow (Binary: Yes/No)

# Challenges
- **Missing Values**: High missing rates in features like Sunshine (48%), Evaporation (43%), and Cloud patterns (38-40%) were addressed by dropping columns exceeding a 30% threshold.
- **Class Imbalance**: The target variable is imbalanced (~76% "No Rain" vs ~22% "Rain"), which was handled using techniques like SMOTE during hyperparameter tuning.

 # Methodology & Preprocessing
1. **Data Cleaning & Feature Engineering**:
 - **Handling Missing Values**: Columns exceeding a 30% missing data threshold (e.g., Sunshine, Evaporation) were removed. Remaining numerical nulls were handled via Median Imputation, while categorical nulls used Mode Imputation.
 - **Encoding**: Categorical variables were transformed using Label Encoding, and the 'Date' feature was separate to year, month, and day.

2. **Data Transformation**
- Outlier Management: Features were capped to reduce the impact of extreme weather anomalies.
- Normalization: Min-Max Scaling was applied to all numerical attributes to ensure feature parity and faster model convergence.
- Data Reshaping: Standard machine learning models typically process data in a 2D format (rows and columns). However, Recurrent Neural Networks (RNNs) require a 3D structure defined as [Samples, Time Steps, Features].
- Oversampling: To address the ~76% "No Rain" class imbalance, RandomOverSampler was implemented to ensure the model learned to identify rainfall events effectively.

# Model Performance & Evaluation

The table below summarizes the performance of the baseline models compared to the final optimized versions. The **Final GRU** model was selected for deployment due to its superior balance of accuracy and recall.

| Model Phase | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Base LSTM** | 84.40% | 71.01% | 48.91% | 57.92% |
| **Base GRU** | 84.77% | 72.82% | 48.91% | 58.50% |
| **Final LSTM (Tuned)** | 77.29% | 49.37% | 78.50% | 60.62% |
| **Final GRU (Tuned)** | **78.43%** | **51.09%** | **79.13%** | **62.13%**
# Key Findings
- **Optimal Architecture**: The GRU model achieved the highest F1-Score, indicating a superior balance between Precision and Recall for this specific dataset.
- **Threshold Optimization**: A custom classification threshold was applied to improve the Recall rate, ensuring that potential rainfall events are not overlooked.
- **Training Efficiency**: GRU layers converged faster than LSTM layers, likely due to the simpler gate mechanism, which requires fewer computational resources.

