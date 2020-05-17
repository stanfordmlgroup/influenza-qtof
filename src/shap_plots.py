import numpy as np

def get_feature_importance(shap_values, X_test, col_names, num_features=10):
    """ Compute feature importance and effect wrt feature mean from
    shap_values and X_test
    Args:
        shap_values {array} -- shap values of test set, shape (num_examples, num_features)
        X_test {array} -- Test data, shape (num_examples, num_features)
        col_names {list} -- Column names of X_test
        num_features {int} -- number of features to plot
    """
    # Determine median value for each feature
    median_value = np.median(X_test, axis=0)
    # Keep where feature contributes positives to classification (shap_values > 0)
    # and the value is greater than the median value
    pos_effect = (shap_values >= 0) * (X_test >= median_value) * (X_test)
    # Percentage high value features contrinute to positive or negative effects
    percent_pos_effect = (pos_effect > 0).sum(axis=0) / pos_effect.shape[0]
    color = percent_pos_effect
    # compute feature importance
    fi_shap = abs(shap_values).sum(0)
    # Normalize
    fi_shap = fi_shap / fi_shap.sum()
    ind = (-fi_shap).argsort()[:num_features]
    return fi_shap[ind], color[ind], col_names[ind]