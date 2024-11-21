import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

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

def HYPER_RandomForest(X_train, y_train, seed : int = 42):
    # Define the model
    rf_model = RandomForestClassifier(random_state=seed)

    # Define the parameter distributions
    param_distributions = {
        'n_estimators': randint(100, 1000),     # Random integers between 100 and 1000
        'max_depth': randint(10, 50),           # Random integers between 10 and 50
        'min_samples_split': randint(2, 20),    # Random integers between 2 and 20
        'min_samples_leaf': randint(1, 10),     # Random integers between 1 and 10
        'max_features': ['sqrt', 'log2', None], # Categorical options
        'bootstrap': [True, False]              # Categorical options
    }

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_distributions,
        n_iter=50,         # Number of random combinations to try
        scoring='balanced_accuracy', # Evaluation metric
        cv=5,              # 5-fold cross-validation
        verbose=2,         # Output progress
        random_state=42,   # For reproducibility
        n_jobs=-1          # Use all available cores
    )
    # Perform the search
    random_search.fit(X_train, y_train)
    # Evaluate on the test set
    return random_search.best_estimator_
