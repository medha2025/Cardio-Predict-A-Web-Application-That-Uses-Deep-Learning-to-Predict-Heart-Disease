# Cardio-Predict-A-Web-Application-That-Uses-Deep-Learning-to-Predict-Heart-Disease
CardioPredict is a web application that uses deep learning to predict heart disease risk from patient medical data. Using the Cleveland Heart Disease dataset, the system performs preprocessing, feature scaling, and neural network training classify patients  having heart disease (1) or not(0), supporting early diagnosis and clinical decision making.
# CardioPredict: A Web Application That Uses Deep Learning to Predict Heart Disease

## Abstract

Heart disease continues to be a major global health concern, emphasizing the importance of early and reliable risk detection. This project presents a deep learning–based approach for predicting heart disease using the Cleveland Heart Disease dataset.

The system integrates **data preprocessing, feature scaling, and a fully connected neural network** trained with optimized hyperparameters and **Early Stopping** to prevent overfitting.

Prior to model development, the dataset is cleaned and processed to handle missing values and convert targets into binary form. The deep learning model is evaluated using multiple performance metrics including **Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix**.

Results demonstrate that the neural network effectively identifies patterns associated with heart disease, offering improved predictive capability compared to traditional approaches.

This work highlights the potential of **deep learning in supporting clinical decision-making**, enabling faster and more accurate diagnosis which may contribute to better patient management and health outcomes.

---

## Keywords

Heart Disease Prediction, Deep Learning, Neural Network, Cleveland Dataset, Data Preprocessing, Standardization, ReLU Activation, Dropout Layers, Early Stopping, Binary Classification, Model Evaluation

---

# System Requirements

## Hardware Requirements

* Desktop/Laptop
* Intel Core i5 processor or above
* Minimum **16 GB RAM**
* 64-bit Operating System

## Software Requirements

| Component            | Technology            |
| -------------------- | --------------------- |
| Operating System     | Windows 11 (64-bit)   |
| IDE                  | Visual Studio Code    |
| Programming Language | Python 3.10.7         |
| Backend              | Flask                 |
| Web Technologies     | HTML, CSS, JavaScript |

## Libraries / Frameworks

* **Scikit-learn** – Machine learning algorithms
* **Pandas** – Data manipulation
* **NumPy** – Numerical computations
* **Matplotlib / Seaborn** – Data visualization
* **TensorFlow / Keras** – Deep learning model development

---

# Dataset

The project uses the **Cleveland Heart Disease Dataset**, which contains medical attributes used to determine whether a patient has heart disease.

The dataset is preprocessed to:

* Handle missing values
* Normalize feature values
* Convert target labels into **binary classification (0 or 1)**

Output:

* **1 → Heart Disease Detected**
* **0 → No Heart Disease**

---

# Proposed Methodology

The system follows the following pipeline:

1. **Data Collection**

   * Cleveland Heart Disease dataset is used.

2. **Data Preprocessing**

   * Handling missing values
   * Encoding categorical features
   * Feature scaling using Standardization

3. **Feature Selection**

   * Important clinical parameters are selected as input features.

4. **Model Development**

   * A **Fully Connected Deep Neural Network** is used.
   * Activation Function: **ReLU**
   * Regularization using **Dropout Layers**
   * Training optimized using **Early Stopping**.

5. **Model Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1 Score
   * ROC-AUC
   * Confusion Matrix

6. **Web Application Integration**

   * The trained model is integrated into a **Flask web application**
   * Users input patient health data
   * The system predicts **heart disease risk**

---

# Input Features Used for Prediction

| Feature Name   | Description                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| Age            | Patient's age in years                                                                               |
| Sex            | Gender (1 = Male, 0 = Female)                                                                        |
| ChestPainType  | Type of chest pain (1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic) |
| RestingBP      | Resting blood pressure (mm Hg)                                                                       |
| Cholesterol    | Serum cholesterol level (mg/dl)                                                                      |
| FastingBS      | Fasting blood sugar >120 mg/dl (1 = Yes, 0 = No)                                                     |
| RestingECG     | Resting electrocardiogram results                                                                    |
| MaxHR          | Maximum heart rate achieved                                                                          |
| ExerciseAngina | Exercise-induced angina (1 = Yes, 0 = No)                                                            |
| Oldpeak        | ST depression induced by exercise                                                                    |
| ST_Slope       | Slope of peak exercise ST segment                                                                    |

---

# Prediction Output

The model predicts whether the patient is at risk of heart disease.

| Output | Meaning                |
| ------ | ---------------------- |
| 1      | Heart Disease Detected |
| 0      | No Heart Disease       |

---

# Technologies Used

* Python
* Flask
* TensorFlow / Keras
* Pandas
* NumPy
* Scikit-learn
* HTML
* CSS
* JavaScript

---

# Installation and Usage

### 1 Clone the Repository

git clone https://github.com/yourusername/CardioPredict.git

### 2 Navigate to Project Folder

cd CardioPredict

### 3 Install Required Libraries

pip install -r requirements.txt

### 4 Run the Application

python app.py

### 5 Open in Browser

http://127.0.0.1:5000

---

# Project Structure

CardioPredict
│
├── dataset
├── models
├── src
├── templates
├── static
├── results
├── requirements.txt
└── README.md

---

# Future Enhancements

* Deploy the application on cloud platforms
* Improve prediction accuracy with larger datasets
* Add real-time hospital data integration
* Develop a mobile-friendly interface

---

# Conclusion

CardioPredict demonstrates how **deep learning techniques can be used to predict heart disease risk with high accuracy**. By combining medical data analysis with a web-based interface, the system provides a practical tool for assisting healthcare professionals in early diagnosis and decision-making.

---
