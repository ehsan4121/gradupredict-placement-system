# GraduPredict

GraduPredict is a single-student final-year project that predicts graduate placement likelihood using Logistic Regression and presents the result through a Streamlit web application.

## Features
- Student data-entry and validation
- Placement probability, outcome, and risk category
- SQLite prediction history
- CSV export
- Dashboard and model metrics
- JSON model format that avoids cross-version pickle errors

## Run locally
1. Create a virtual environment.
2. Install packages: `pip install -r requirements.txt`
3. Train or refresh the model: `python train_model.py`
4. Start the app: `streamlit run app.py`

## Dataset
`data/placement_demo.csv` is a reproducible schema-compatible demonstration dataset so the package runs immediately. For the final academic experiment, download a Campus Recruitment / Placement Prediction dataset from Kaggle, map its columns to the feature list in `train_model.py`, document the source and license, then retrain. Do not report the demonstration metrics as Kaggle results.

## Main files
- `app.py`: Streamlit application
- `prediction.py`: compatible model loader and prediction logic
- `train_model.py`: NumPy Logistic Regression training
- `model/placement_model.json`: trained model
- `database/gradupredict.db`: SQLite database
- `notebook.ipynb`: step-by-step experiment
- `docs/`: SRS and design documents
- `screenshots/`: final interface mockups
