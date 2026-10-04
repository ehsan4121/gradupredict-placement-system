import sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from prediction import load_model,predict_probability,classify
def test_prediction_range():
 m=load_model(); values={f: m["mean"][i] for i,f in enumerate(m["features"])}; p=predict_probability(values,m); assert 0<=p<=1
def test_classification():
 assert classify(.8)==("Likely placed","Low"); assert classify(.2)==("At risk","High")
