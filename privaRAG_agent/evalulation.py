# privaRAG_agent/evaluation.py
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return {"precision": prec, "recall": rec, "f1": f1}
