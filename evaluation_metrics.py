import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, classification_report

global results
results = pd.DataFrame({})


def evaluate_models(model_name, result, y_test, y_pred):
    """

    :param model_name:
    :param result:
    :param y_test:
    :param y_pred:
    :return:

    :UC: y_test must be a Pandas Series with a label
    """
    print("Prediction for : ", y_test.name)  # name Pandas Series
    f1_score_positif = f1_score(y_test, y_pred, average='binary').round(3)
    f1_score_weighted = f1_score(y_test, y_pred, average='weighted').round(3)
    recall = recall_score(y_test, y_pred).round(3)
    precision = precision_score(y_test, y_pred).round(3)
    accuracy = accuracy_score(y_test, y_pred).round(3)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate).round(3)

    result = pd.concat([result, pd.DataFrame({"Model": [model_name],
                                              "F1-score": [f1_score_positif],
                                              "F1-score weighted": [f1_score_weighted],
                                              "Precision": [precision],
                                              "Recall": [recall],
                                              "Accuracy": [accuracy],
                                              "ROC-AUC": [roc_auc],
                                              })])
    # we sort the datafraeme of results by best : by=["F1-score]
    result = result.sort_values(by=["F1-score"], ascending=False)
    display(result)

    return result


def confusion(y_test, y_pred):
    """
    Displays a fancy confusion matrix
    :param y_test:
    :param y_pred:
    :return:
    """
    mat = confusion_matrix(y_test, y_pred)  # a numpy array
    mat = pd.DataFrame(mat)
    mat.columns = [f"pred_{i}" for i in mat.columns]
    mat.index = [f"test_{i}" for i in mat.index]

    return mat


def evaluate_classification(y_test, y_pred, y_pred_proba):
    """

    :param y_test:
    :param y_pred:
    :param y_pred_proba:
    :return:
    """
    # 1) Metrics
    print(classification_report(y_test, y_pred))

    # 2) Confusion matrix
    conf_mat = confusion(y_pred, y_test)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='g')  # fmt='g' to get the numbers not in scientific notation

    # 3) ROC
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate).round(3)
    print("ROC-AUC", roc_auc)

    plt.figure()
    plt.title("Receiver Operating Characteristic")
    plt.plot(false_positive_rate, true_positive_rate, color="red", label="AUC = %0.2f" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.axis("tight")
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()

    # 4) Sensibility Specificity
    [fpr, tpr, thr] = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, color='coral', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - specificite', fontsize=14)
    plt.ylabel('Sensibilite', fontsize=14)


def score(estimator, X_train, X_test, y_train, y_test):  ## To delete if not needed : for accuracy
    """
    Computes and prints train score and test score.
    :param estimator:
    :return:
    """
    tr_score = estimator.score(X_train, y_train).round(4)
    te_score = estimator.score(X_test, y_test).round(4)

    print(f"score train : {tr_score} score test : {te_score} ")
