# Kaggle Dataset Integration

The included `placement_demo.csv` makes the project runnable immediately, but its metrics must be described as demonstration results. Complete these steps before reporting Kaggle findings.

1. Download a Campus Recruitment or Placement Prediction CSV from Kaggle and retain its dataset title, author, URL, license, and download date in the final report.
2. Store the untouched file in `data/raw/` and do not overwrite it.
3. Create a cleaned file whose columns map to the model features in `train_model.py`.
4. Convert Yes and No columns to 1 and 0. Convert the placement target to 1 for placed and 0 for not placed.
5. Remove direct identifiers, salary, and any field that becomes available only after placement.
6. Check missing values, duplicates, impossible percentages, class balance, and target leakage.
7. Split the data before calculating means and standard deviations. Fit all preprocessing with the training partition only.
8. Update the CSV path in the notebook and training script, then run the complete notebook from the first cell.
9. Replace the demonstration model JSON only after reviewing the held-out accuracy, precision, recall, F1 score, and confusion matrix.
10. Update the SRS, design document, screenshots, and presentation only where actual Kaggle results differ.

## Suggested column mapping

| Project feature | Possible dataset source |
|---|---|
| `ssc_percentage` | Secondary-school percentage |
| `hsc_percentage` | Higher-secondary percentage |
| `degree_percentage` | Degree percentage |
| `employability_score` | Employability or placement-test percentage |
| `work_experience` | Work-experience Yes or No field |
| `placed` | Placement-status target |

If the selected dataset does not contain internships, projects, communication, technical ratings, backlogs, or training, remove those features consistently from the CSV, notebook, training script, prediction form, JSON model, and documentation. Do not invent unavailable values.

