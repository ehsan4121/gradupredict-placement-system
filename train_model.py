from pathlib import Path
import json, numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent
FEATURES=['ssc_percentage', 'hsc_percentage', 'degree_percentage', 'employability_score', 'internships', 'projects', 'communication_rating', 'technical_rating', 'backlogs', 'work_experience', 'training_completed']
df=pd.read_csv(ROOT/"data"/"placement_demo.csv"); X=df[FEATURES].to_numpy(float); y=df["placed"].to_numpy(float)
rng=np.random.default_rng(610); idx=rng.permutation(len(y)); cut=int(.8*len(y)); tr,te=idx[:cut],idx[cut:]
mean=X[tr].mean(0); scale=X[tr].std(0); scale[scale==0]=1; xs=(X[tr]-mean)/scale; w=np.zeros(X.shape[1]); b=0.0
for _ in range(1800):
 p=1/(1+np.exp(-(xs@w+b))); e=p-y[tr]; w-=.08*(xs.T@e/len(tr)); b-=.08*e.mean()
probs=1/(1+np.exp(-(((X[te]-mean)/scale)@w+b))); pred=(probs>=.5); acc=float((pred==y[te]).mean())
tp=int(((pred==1)&(y[te]==1)).sum()); fp=int(((pred==1)&(y[te]==0)).sum()); fn=int(((pred==0)&(y[te]==1)).sum())
precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); f1=2*precision*recall/max(precision+recall,1e-9)
model={"model_type":"Logistic Regression implemented with NumPy","features":FEATURES,"mean":mean.tolist(),"scale":scale.tolist(),"weights":w.tolist(),"intercept":float(b),"threshold":.5,"metrics":{"accuracy":acc,"precision":precision,"recall":recall,"f1":f1},"training_source":"Included schema-compatible demonstration dataset. Retrain with the cited Kaggle dataset for academic reporting."}
(ROOT/"model"/"placement_model.json").write_text(json.dumps(model,indent=2)); print(f"Accuracy: {acc:.3f}; F1: {f1:.3f}")
