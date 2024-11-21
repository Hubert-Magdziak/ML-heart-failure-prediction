import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

def ROC(y_true, y_pred_proba, return_optimal_threshold : bool = False):
    fpr, tpr, threshold = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    if return_optimal_threshold:
    # Find optimal cut-off point
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = threshold[optimal_idx]
        print("Threshold value is:", optimal_threshold)
        return optimal_threshold
    # ROC plot
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()