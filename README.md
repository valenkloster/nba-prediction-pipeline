# 🏀 NBA Prediction Pipeline

This project builds a complete pipeline for cleaning, transforming, and engineering features from NBA game data, preparing it for machine learning models.

---

## ⚙️ Setup Instructions

### 1. Create a virtual environment
```bash
python3 -m venv venv
```

### 2. Activate the environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

---

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

### 4. Create the folder structure

Inside the project root, create the following directories:

```
data/
│
├── raw/         # Original CSV files (Games.csv, Players.csv, etc.)
├── processed/   # Intermediate cleaned datasets
└── final/       # Final dataset ready for modeling
```

You can download the raw CSVs separately (they are not uploaded to GitHub due to size limits).

---

### 5. Run the notebooks

Execute the following notebooks in order:

1. `notebooks/01_data_cleaning.ipynb`  
   → Cleans raw CSV files, standardizes formats, handles missing data.

2. `notebooks/02_feature_engineering.ipynb`  
   → Builds target variables, team differentials, rolling performance indicators, and exports the final dataset.

---

## 🧠 Next Steps

After preparing the final dataset, you can continue with:

- `notebooks/03_model_training.ipynb`
- `notebooks/04_model_evaluation.ipynb`
- `notebooks/05_model_backtesting.ipynb`

*(These are included as references and should be updated to reflect the current cleaned dataset.)*
