🩺 Disease Predictor - A Machine Learning Web App
This is a web application that predicts the likelihood of having Diabetes or Anemia based on user-input health data. The project uses machine learning models to make predictions and a Flask-based backend to serve the user interface.

✨ Features
Dual Disease Prediction: Predicts the probability of two different health conditions:

Diabetes

Anemia

User-Friendly Web Interface: Simple and intuitive HTML forms for data input.

Real-Time Predictions: Instantly get a prediction after submitting the required health parameters.

Powered by Scikit-learn: Utilizes robust machine learning models trained for classification tasks.

🛠️ Technologies & Libraries Used
The project is built using the following technologies:

Backend: Python, Flask

Frontend: HTML, CSS

Machine Learning: Scikit-learn, Pandas, NumPy

Development Environment: Jupyter Notebook

🚀 Setup and Installation
To run this project on your local machine, please follow these steps.

Prerequisites
Python 3.7+

pip (Python package installer)

Installation Steps
Clone the repository:

Bash

git clone https://github.com/your-username/disease-predictor.git
Navigate into the project directory:

Bash

cd disease-predictor
Create and activate a virtual environment (recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required dependencies:

Bash

pip install Flask pandas scikit-learn
Run the Flask application:

Bash

python app.py
Open your web browser and navigate to http://127.0.0.1:5000/ to use the application.

📂 File Structure
The project directory is organized as follows:

/disease-predictor
|
|-- templates/
|   |-- about.html              # About page for the project
|   |-- anemia.html             # Input form for Anemia prediction
|   |-- diabetes.html           # Input form for Diabetes prediction
|   |-- disease_prediction.html # Main homepage
|
|-- app.py                      # The main Flask application file
|-- Multiple_Disease_Prediction.ipynb # Jupyter notebook for model training
|-- Blood_samples_dataset.csv   # Dataset used for training
|-- svm_model.pkl               # Trained SVM model for Diabetes prediction
|-- knn_model.pkl               # Trained KNN model for Anemia prediction
|-- scaling.pkl                 # Saved scaler for data preprocessing
|-- ... (other static files like CSS, images)
🤖 Machine Learning Models
This application uses two distinct machine learning models for its predictions:

Diabetes Prediction: A Support Vector Machine (SVM) model (svm_model.pkl) is used to classify whether a person has diabetes based on metrics like glucose level, blood pressure, BMI, etc.

Anemia Prediction: A K-Nearest Neighbors (KNN) model (knn_model.pkl) is used to determine the likelihood of anemia based on blood sample data such as Gender, Hemoglobin, MCH, MCV, and MCHC.

The complete model training and data exploration process can be found in the Multiple_Disease_Prediction.ipynb Jupyter Notebook.
