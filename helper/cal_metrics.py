from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    root_mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
    )

from scipy.stats import (
    pearsonr,
    spearmanr
)

def classification_metrics(y_true, y_pred, y_score):
    metrics = {}
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics['accuracy'] = report['accuracy']
    metrics['recall'] = report['weighted avg']['recall']
    metrics['precision'] = report['weighted avg']['precision']
    metrics['f1-score'] = report['weighted avg']['f1-score']
    for lbl in ("0", "0.0"):
        if lbl in report:
            metrics["0_recall"] = report[lbl]["recall"]
            metrics["0_precision"] = report[lbl]["precision"]
            break
    for lbl in ("1", "1.0"):
        if lbl in report:
            metrics["1_recall"] = report[lbl]["recall"]
            metrics["1_precision"] = report[lbl]["precision"]
            break
    metrics['auc-roc'] = roc_auc_score(y_true, y_score, average='weighted')
    metrics['auc-prc'] = average_precision_score(y_true, y_score, average='weighted')

    return metrics

def regression_metrics(y_true, y_pred):
    metrics = {}
    metrics['rmse'] = root_mean_squared_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    metrics['pearsonr'] = pearsonr(y_true, y_pred).statistic
    metrics['spearmanr'] = spearmanr(y_true, y_pred).statistic

    return metrics