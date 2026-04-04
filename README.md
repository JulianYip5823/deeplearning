# Rainfall Prediction in Australia using Deep Learning
This repository contains the implementation of a deep learning project/assignment aimed at predicting rainfall in Australia using historical weather data. This project was developed as part of an Individual assignment for the Optimization and Deep Learning module. 

## Project Overview
Accurate rainfall prediction is critical for sectors like agriculture and water management in Australia. This research explores the effectiveness of Recurrent Neural Networks (RNNs)-specifically Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU)-in capturing temporal dependencies within weather data to predict whether it will rain the following day ("RainTomorrow")

### Objectives
- **Data Preparation**: Perform rigorous cleaning, handle missing values, and normalize features for high-quality model input
- **Exploratory Data Analysis (EDA)**: Identify distributions and relationships between weather characteristics and rainfall
- **Model Implementation**: Build and compare two RNN architectures: LSTM and GRU
- **Model Optimization**: Utilize hyperparameter tuning (e.g., Keras Tuner) to optimize factors like learning rate and dropout
- **Evaluation**: Assess performance using Accuracy, Precision, Recall, and F1-score

# Dataset
The project utilizes the "Rain in Australia" dataset (sourced from the Australian Bureau of Meteorology via Kaggle).
- **Size**: approximately 145,460 observations.
- **Dimensions**: 23 features, including temperature, humidity, pressure, and wind patterns.
- **Target Variable**: RainTomorrow (Binary: Yes/No)

# Challenges
- **Missing Values**: High missing rates in features like Sunshine (48%), Evaporation (43%), and Cloud patterns (38-40%) were addressed by dropping columns exceeding a 30% threshold.
- **Class Imbalance**: The target variable is imbalanced (~76% "No Rain" vs ~22% "Rain"), which was handled using techniques like SMOTE during hyperparameter tuning.

 # Methodology & Preprocessing
 
