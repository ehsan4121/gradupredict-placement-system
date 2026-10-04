# GraduPredict Final Viva Demonstration

## Before the viva

1. Open a terminal in the project folder.
2. Install the packages with `pip install -r requirements.txt`.
3. Run `python train_model.py` and record the printed accuracy and F1 score.
4. Start the interface with `streamlit run app.py`.
5. Confirm that the Dashboard opens and that the database file is writable.

## Live demonstration sequence

1. **Dashboard:** Explain total predictions, outcome counts, recent activity, and model accuracy.
2. **New Prediction:** Enter a student ID and academic and employability values.
3. **Validation:** Explain that bounded controls prevent impossible values and a blank student ID is rejected.
4. **Prediction:** Select **Predict Placement** and explain probability, predicted outcome, risk, and the decision threshold.
5. **Responsible use:** Read the note explaining that the estimate supports counseling and does not decide hiring eligibility.
6. **Save:** Select **Save Prediction** and confirm that the record was written to SQLite.
7. **History:** Filter the records and download the displayed data as CSV.
8. **About Model:** Explain Logistic Regression, standardization, held-out evaluation, JSON model storage, and the dataset limitation.

## Short technical explanation

The training script divides the data into training and test partitions. It calculates the mean and standard deviation from the training partition, standardizes each input, and learns Logistic Regression weights by gradient descent. Runtime prediction loads the feature order, scaling values, weights, intercept, threshold, and metrics from JSON. SQLite stores only the final prediction record.

## Evidence to show the examiner

- `notebook.ipynb` for the experiment workflow
- `model/placement_model.json` for the trained parameters
- `database/gradupredict.db` for saved records
- `tests/test_prediction.py` for automated checks
- `docs/GraduPredict_SRS.docx` and `docs/GraduPredict_Design_Document.docx`
- `screenshots/` for the expected interface sequence

