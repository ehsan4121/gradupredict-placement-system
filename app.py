"""GraduPredict single-student Streamlit demonstration."""

from pathlib import Path
import sqlite3
import pandas as pd
import streamlit as st
from prediction import classify, load_model, predict_probability

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "database" / "gradupredict.db"
MODEL_PATH = APP_DIR / "model" / "placement_model.json"

st.set_page_config(page_title="GraduPredict", page_icon="🎓", layout="wide")

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

def init_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT NOT NULL,
            prediction TEXT NOT NULL, probability REAL NOT NULL,
            risk TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)""")

def save_prediction(student_id, prediction, probability, risk):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            "INSERT INTO predictions(student_id,prediction,probability,risk) VALUES(?,?,?,?)",
            (student_id, prediction, probability, risk),
        )

def load_history():
    with sqlite3.connect(DB_PATH) as connection:
        return pd.read_sql_query(
            "SELECT id,student_id,prediction,probability,risk,created_at "
            "FROM predictions ORDER BY id DESC", connection
        )

init_database()
st.title("GraduPredict")
st.subheader("Graduate placement prediction with Logistic Regression")
st.write(
    "Enter academic and employability information for one student. The system "
    "uses the same eleven features as the included training notebook."
)

with st.expander("How to interpret the result"):
    st.write(
        "The model returns an estimated placement probability. A value at or above "
        "the selected threshold is labelled **Likely placed**. Risk is Low at 70% "
        "or above, Medium from 45% to below 70%, and High below 45%."
    )
    st.info(
        "This academic prototype supports student counseling and must not decide "
        "whether a person is eligible for employment."
    )

try:
    model = get_model()
except Exception as exc:
    st.error(f"The saved model could not be loaded. {exc}")
    st.stop()

with st.form("placement_prediction_form"):
    st.markdown("### Student information")
    col1, col2, col3 = st.columns(3)
    with col1:
        student_id = st.text_input("Student ID", value="GP-1043")
        ssc_percentage = st.number_input("Secondary-school percentage", 0.0, 100.0, 70.0, 0.5)
        hsc_percentage = st.number_input("Higher-secondary percentage", 0.0, 100.0, 74.0, 0.5)
        degree_percentage = st.number_input("Degree percentage", 0.0, 100.0, 72.5, 0.5)
    with col2:
        employability_score = st.number_input("Employability test score", 0.0, 100.0, 78.0, 0.5)
        internships = st.number_input("Internships completed", 0, 10, 1)
        projects = st.number_input("Projects completed", 0, 20, 3)
        backlogs = st.number_input("Current or previous backlogs", 0, 20, 0)
    with col3:
        communication_rating = st.slider("Communication rating", 1, 5, 4)
        technical_rating = st.slider("Technical-skill rating", 1, 5, 4)
        work_experience = st.selectbox("Previous work experience", ["No", "Yes"])
        training_completed = st.selectbox("Employability training completed", ["No", "Yes"], index=1)
    threshold = st.slider(
        "Placement decision threshold", 0.30, 0.70, 0.50, 0.05,
        help="The project evaluation uses 0.50. Changing it is for demonstration only."
    )
    save_result = st.checkbox("Save this prediction to history", value=True)
    submitted = st.form_submit_button("Predict placement likelihood", type="primary")

if submitted:
    if not student_id.strip():
        st.error("Student ID is required before generating a prediction.")
    else:
        values = {
            "ssc_percentage": ssc_percentage, "hsc_percentage": hsc_percentage,
            "degree_percentage": degree_percentage, "employability_score": employability_score,
            "internships": internships, "projects": projects,
            "communication_rating": communication_rating, "technical_rating": technical_rating,
            "backlogs": backlogs, "work_experience": work_experience == "Yes",
            "training_completed": training_completed == "Yes",
        }
        try:
            probability = predict_probability(values, model)
            prediction, risk = classify(probability, threshold)
        except Exception as exc:
            st.error(f"Prediction failed. {exc}")
        else:
            st.markdown("### Prediction")
            left, right = st.columns([1, 2])
            with left:
                st.metric("Estimated placement probability", f"{probability:.1%}")
                st.metric("Risk category", risk)
            with right:
                if prediction == "Likely placed":
                    st.success(f"Likely placed: probability is at or above the {threshold:.0%} threshold.")
                else:
                    st.warning(f"At-risk student: probability is below the {threshold:.0%} threshold.")
                st.write("The estimate reflects the academic, project, training, and experience values entered above.")
            st.progress(min(max(probability, 0.0), 1.0))
            st.caption("Use this result for academic demonstration and counseling only.")
            if save_result:
                save_prediction(student_id.strip(), prediction, probability, risk)
                st.toast("Prediction saved to history.", icon="✅")
            with st.expander("Model input row"):
                st.dataframe(pd.DataFrame([{"student_id": student_id.strip(), **values}]), width="stretch", hide_index=True)

st.divider()
dashboard_tab, history_tab, model_tab = st.tabs(["Dashboard", "Prediction history", "About the model"])
history = load_history()

with dashboard_tab:
    total = len(history)
    likely = int((history["prediction"] == "Likely placed").sum()) if total else 0
    at_risk = int((history["prediction"] == "At risk").sum()) if total else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Saved predictions", total)
    c2.metric("Likely placed", likely)
    c3.metric("At risk", at_risk)
    c4.metric("Model accuracy", f"{model.get('metrics', {}).get('accuracy', 0):.1%}")
    if total:
        st.bar_chart(history["prediction"].value_counts().rename_axis("Outcome").to_frame("Count"))
        st.dataframe(history.head(5), width="stretch", hide_index=True)
    else:
        st.info("Saved predictions will appear here after the first demonstration.")

with history_tab:
    outcome_filter = st.selectbox("Filter prediction history", ["All", "Likely placed", "At risk"])
    displayed = history if outcome_filter == "All" else history[history["prediction"] == outcome_filter]
    shown = displayed.copy()
    if not shown.empty:
        shown["probability"] = shown["probability"].map(lambda value: f"{value:.1%}")
    st.dataframe(shown, width="stretch", hide_index=True)
    st.download_button("Download displayed history as CSV", displayed.to_csv(index=False), "gradupredict_history.csv", "text/csv", disabled=displayed.empty)

with model_tab:
    st.write("**Algorithm:** Logistic Regression implemented with NumPy")
    st.write("**Target:** Likely placed or At risk")
    metric_columns = st.columns(4)
    for column, key in zip(metric_columns, ["accuracy", "precision", "recall", "f1"]):
        column.metric(key.title(), f"{model.get('metrics', {}).get(key, 0):.1%}")
    st.write(
        "The included CSV is a reproducible demonstration dataset. Replace it with "
        "the documented Kaggle campus-placement dataset before reporting final findings."
    )

st.divider()
st.caption("GraduPredict | Campus placement demonstration | Compatible JSON Logistic Regression")
