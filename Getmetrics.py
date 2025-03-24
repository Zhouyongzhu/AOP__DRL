from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_curve, auc, average_precision_score, f1_score, precision_score
import torch
import numpy as np
import torch.nn as nn

def getMetrics(eval_preds):
    logits, y_true = eval_preds
    sm = nn.Softmax(dim=1)
    y_proba = sm(torch.tensor(logits))
    y_proba = np.array(y_proba)
    
    # 计算预测标签
    logits01 = np.argmax(y_proba, axis=-1).flatten()
    y_pred = logits01.astype(np.int16)

    # 计算评估指标
    ACC = accuracy_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    CM = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = CM.ravel()
    Sn = tp/(tp+fn)  # 灵敏度（Sensitivity）
    Sp = tn/(tn+fp)  # 特异性（Specificity）
    Precision = precision_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred)
    FPR, TPR, thresholds_ = roc_curve(y_true, y_proba[:, 1])
    AUC = auc(FPR, TPR)
    AUPR = average_precision_score(y_true, y_proba[:, 1])

    # 返回字典，包含 `eval_accuracy` 和其他指标
    return {
        'eval_accuracy': ACC,  # 必须有 eval_accuracy
        'eval_mcc': MCC,  # 马修斯相关系数
        'eval_sensitivity': Sn,  # 灵敏度
        'eval_specificity': Sp,  # 特异性
        'eval_precision': Precision,  # 精确率 recall
        'eval_f1': F1,  # F1分数
        'eval_auc': AUC,  # AUC
        'eval_aupr': AUPR  # AUPR
    }

def getScore(logits):
    sm = nn.Softmax(dim=1)
    y_proba = sm(torch.tensor(logits))
    y_proba = np.array(y_proba)
    return y_proba

def getPredictLabel(logits):
    sm = nn.Softmax(dim=1)
    y_proba = sm(torch.tensor(logits))
    y_proba = np.array(y_proba)
    logits01 = np.argmax(y_proba, axis=-1).flatten()
    y_pred = logits01.astype(np.int16)
    return y_pred
