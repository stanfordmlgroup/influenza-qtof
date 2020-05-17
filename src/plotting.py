# Model specific plots
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn import metrics
import delong


def plot_random_forest_hp_optimization(models):
    """ Plot random forest hyperparameter optimization

    Arguments:
        models {List} -- List of trained CV models to plot
    """
    n_estimators = [i.params_dict['n_estimators'] for i in models]
    min_samples_split = [i.params_dict['min_samples_split'] for i in models]
    mean_auc = [i.cv_storage.get_mean_auc() for i in models]

    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    sc = ax.scatter(n_estimators, min_samples_split, c=mean_auc, cmap='RdYlGn')
    fig.colorbar(sc)
    plt.title("Mean Cross-Validation AUC as a function of n_estimators and min_samples_split")
    plt.xlabel('n_estimators')
    plt.ylabel('min_samples_split')

def plot_lgbm_hp_optimization(models):
    """ Plot random forest hyperparameter optimization

    Arguments:
        models {List} -- List of trained CV models to plot
    """
    max_depth = [i.params_dict['max_depth'] for i in models]
    min_samples_split = [i.params_dict['min_child_samples'] for i in models]
    mean_auc = [i.cv_storage.get_mean_auc() for i in models]

    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    sc = ax.scatter(max_depth, min_samples_split, c=mean_auc, cmap='RdYlGn')
    fig.colorbar(sc)
    plt.title("Mean Cross-Validation AUC as a function of n_estimators and min_samples_split")
    plt.xlabel('max_depth')
    plt.ylabel('min_samples_split')

def plot_roc_curve(fpr, tpr, auc, title):
    fig, ax = plt.subplots(1,1, figsize=(8, 8))

    ax.plot([0,1], [0,1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.plot(fpr, tpr, label='ROC Test Set (AUC = %0.2f)' % (auc))

    plt.legend(loc='lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)


def plot_roc_curve_folds(fprs, tprs, aucs, title):
    """ Plot ROC curves of kfolds of cross val

    Arguments:
        fprs - list of fpr for each fold (list of lists)
        tprs - list of tpr (list of lists)
        auc - list of auc
        title - title of graph

    Returns:
        mean values of each
    """
    # Copy these to avoid changing original data
    fprs = fprs.copy()
    tprs = tprs.copy()
    aucs = aucs.copy()

    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(8,8))
    for i in range(len(fprs)):
        roc_auc = aucs[i]
        plt.plot(fprs[i], tprs[i], lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        tprs[i] = interp(mean_fpr, fprs[i], tprs[i])
        tprs[i][0] = 0.0

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)


    plt.legend(loc='lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()

    return mean_fpr, mean_tpr, mean_auc

def plot_roc_curve_folds_with_test_roc(fprs, tprs, aucs, test_fpr, test_tpr, test_auc, title):
    """ Plot ROC curves of kfolds of cross val

    Arguments:
        fprs - list of fpr for each fold (list of lists)
        tprs - list of tpr (list of lists)
        auc - list of auc
        title - title of graph

    Returns:
        mean values of each
    """
    # copy these to avoid changing values in original list
    fprs = fprs.copy()
    tprs = tprs.copy()
    aucs = aucs.copy()
    test_fpr = test_fpr.copy()
    test_tpr = test_tpr.copy()
    test_auc = test_auc.copy()

    mean_fpr = np.linspace(0, 1, 100)
    fig = plt.figure(figsize=(8,8))
    for i in range(len(fprs)):
        roc_auc = aucs[i]
        plt.plot(fprs[i], tprs[i], lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        tprs[i] = interp(mean_fpr, fprs[i], tprs[i])
        tprs[i][0] = 0.0

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    plt.plot(test_fpr, test_tpr, color='g',
        label='Test ROC (AUC = %0.2f)' % (test_auc),
        lw=2, alpha=1
    )

    plt.legend(loc='lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()

    return mean_fpr, mean_tpr, mean_auc, fig

def plot_multiclass_roc(y_true, y_pred, label_sets, legend_names, title):
    """ Plot ROC curves for the subproblems in label_sets
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot([0,1], [0,1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    for label_set, name in zip(label_sets, legend_names):
        pos = np.array(y_pred[:, label_set[0]].sum(axis=1))
        pos_label = np.array([1 if i in label_set[0] else 0 for i in y_true])
        print(len(pos), len(pos_label))
        fpr, tpr, _ = roc_curve(pos_label, pos)
        print(len(fpr), len(tpr))
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (name, a))

        print(name)
        print(delong.compute_stats(.95, pos, pos_label))
        print()

    plt.legend(loc='lower right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()

    return fig
