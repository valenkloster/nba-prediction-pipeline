# 🏀 NBA Game Outcome Predictor

This project is a complete end-to-end data science pipeline that predicts the outcome of NBA basketball games. It fetches historical game data, engineers predictive features, trains two machine learning models, and serves the predictions through an interactive web application built with Gradio.



## ✨ Features

-   **Data Pipeline:** A series of Jupyter notebooks that handle data cleaning, feature engineering, and model training.
-   **Dual-Model Prediction:**
    1.  **Win/Loss Classifier:** A Random Forest model that predicts the probability of the home team winning.
    2.  **Score Difference Regressor:** A Random Forest model that predicts the final score difference.
-   **Time-Based Validation:** The models are trained on past data and evaluated on more recent data to simulate real-world prediction scenarios and prevent data leakage.
-   **Interactive Web App:** A user-friendly interface built with Gradio that allows anyone to select two teams and get an instant game prediction.

## 📊 Model Performance

-   **Win/Loss Classifier Accuracy:** **57.2%** on the test set.
-   **Score Difference Regressor MAE:** **11.35 points** (Mean Absolute Error).

This means the model correctly predicts the winner more often than not, and its score difference prediction is, on average, off by about 11 points.

## 🛠️ Technology Stack

-   **Python:** The core programming language.
-   **Pandas:** For data manipulation and analysis.
-   **Scikit-learn:** For building and training the machine learning models.
-   **Joblib:** For saving and loading the trained models.
-   **Gradio:** For creating the interactive web application.
-   **Conda:** For environment management.
-   **Git & Git LFS:** For version control and handling large model files.

## 📂 Project Structure

```
nba-prediction-pipeline/
├── 📁 data/              # Contains raw, processed, and final datasets.
├── 📁 models/            # Stores the saved .pkl model files.
├── 📁 notebooks/         # Jupyter notebooks for the data pipeline.
│   ├── 01_data_cleaning.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── 🐍 app.py             # The Python script for the Gradio web application.
├── .gitattributes       # Configures Git LFS to track .pkl files.
└── README.md            # You are here!
```

## 🚀 How to Run the Application

To run this project on your local machine, please follow these steps.

### 1. Clone the Repository

First, clone the repository to your local machine.
```bash
git clone https://github.com/valenkloster/nba-prediction-pipeline.git
cd nba-prediction-pipeline
```

### 2. Set Up the Environment

This project uses Conda for environment management.

```bash
# Create a new conda environment named 'nba_predictor'
conda create --name nba_predictor python=3.10 -y

# Activate the new environment
conda activate nba_predictor

# Install the required packages
# (Note: This could also be done via a requirements.txt or environment.yml)
conda install -c conda-forge pandas scikit-learn jupyterlab gradio git-lfs -y
```

### 3. Set Up Git LFS

The trained models are stored using Git Large File Storage (LFS). You need to pull these files from LFS storage.

```bash
# Install Git LFS in your local repo (one-time setup)
git lfs install

# Pull the large files from the remote storage
git lfs pull
```
After this step, you should see the `win_loss_classifier.pkl` and `score_diff_regressor.pkl` files inside your `models` folder.

### 4. Launch the App

You are now ready to run the application!

```bash
python app.py
```

Open your web browser and navigate to the local URL provided in the terminal (usually `http://127.0.0.1:7860`).

---
*This project was developed as a comprehensive exercise in building a real-world machine learning pipeline.*
