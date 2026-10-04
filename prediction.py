import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def load_model(path=ROOT/"model"/"placement_model.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def predict_probability(values, model=None):
    model=model or load_model()
    x=np.array([float(values[f]) for f in model["features"]])
    z=((x-np.array(model["mean"]))/np.array(model["scale"]))@np.array(model["weights"])+model["intercept"]
    return float(1/(1+np.exp(-z)))

def classify(probability, threshold=.5):
    prediction="Likely placed" if probability>=threshold else "At risk"
    risk="Low" if probability>=.70 else "Medium" if probability>=.45 else "High"
    return prediction,risk
