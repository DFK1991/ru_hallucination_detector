from sklearn.metrics import f1_score, accuracy_score
import numpy as np

def compute_metrics(eval_pred, dict_idx2label):
    
    labels_list = list(dict_idx2label.keys())

    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    f1_micro = f1_score(labels, preds, average="micro")
    
    f1_per_class = f1_score(labels, preds, average=None, labels=labels_list, zero_division=0)
    
    metrics = {"accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_micro": f1_micro,
              }
    
    for i, label_name in dict_idx2label.items():
        metrics[f"f1_{label_name}"] = f1_per_class[i]
        
    return metrics
