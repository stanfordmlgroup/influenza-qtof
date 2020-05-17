""" Useful functions for model selection/cross validation
"""
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score
import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import models
SEED=2019

class HyperparameterSearchResults():
    def __init__(self):
        self.models = []

    def add_model(self, model):
        """ Add a model to results
        """
        self.models.append(model)

    def get_all_models(self):
        """ Return all trained models
        """
        return self.models

    def get_best_model(self):
        """ Gets the best model based on mean AUC across folds
        """
        mean_aucs = [i.cv_storage.get_mean_auc() for i in self.models]
        max_auc_idx = np.argmax(mean_aucs)
        return self.models[max_auc_idx]

def cv_hyperparameter_opt(train_df, models, n_splits, multiclass=False):
    """ Hyperparameter
    
    Arguments:
        train_df {pd.DataFrame} -- Pandas dataframe of training data
        models {List} -- List of models extending the Model class
        n_splits {Int} -- number of splits for each CV
    """
    hp_search_results = HyperparameterSearchResults()
    total_models = len(models)
    for idx, model in enumerate(models):
        if idx % 10 == 0:
            print("Training model {} of {}".format(idx, total_models))

        if multiclass:
            model = _run_cv_train_multiclass(train_df, model, n_splits)
        else:
            model = _run_cv_train(train_df, model, n_splits)
        hp_search_results.add_model(model)
    return hp_search_results

def _run_cv_train(train_df, model, n_splits):
    """ Runs cross-validation training using the model
    
    Arguments:
        train_df {pd.DataFrame} -- Pandas dataframe of training data
        model {Model} -- Model class. Has a cv_results attribute
        n_splits {int} -- nubmer of splits for cross validation
    """
    cv = StratifiedKFold(n_splits=n_splits, random_state=SEED)
    X, y = utils.df_to_array(train_df)

    for train, test in cv.split(X, y):
        clf = model.init_model(model.model_name, model.params_dict)
        clf.fit(X[train], y[train])
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(X[test])
            fpr, tpr, _ = roc_curve(y[test], proba[:, 1])
            cv_auc = auc(fpr, tpr)
        else:
            proba = clf.decision_function(X[test])
            fpr, tpr, _ = roc_curve(y[test], proba)
            cv_auc = auc(fpr, tpr)

        model.cv_storage.auc.append(cv_auc)
        model.cv_storage.fprs.append(fpr)
        model.cv_storage.tprs.append(tpr)
        model.cv_storage.model.append(clf)

    return model

def _run_cv_train_multiclass(train_df, model, n_splits):
    """ Runs cross-validation training using the model
    
    Arguments:
        train_df {pd.DataFrame} -- Pandas dataframe of training data
        model {Model} -- Model class. Has a cv_results attribute
        n_splits {int} -- nubmer of splits for cross validation
    """
    cv = StratifiedKFold(n_splits=n_splits)
    X, y = utils.df_to_array(train_df)
    
    for train, test in cv.split(X, y):
        clf = model.init_model('', model.params_dict)
        clf.fit(X[train], y[train])
        proba = clf.predict_proba(X[test])

        # make one hot multiclass labels
        y_one_hot = np.zeros(proba.shape)
        for idx, label in enumerate(y[test]):
            y_one_hot[idx, label] = 1

        fpr, tpr, _ = roc_curve(y_one_hot.ravel(), proba.ravel())
        cv_auc = auc(fpr, tpr)

        model.cv_storage.auc.append(cv_auc)
        model.cv_storage.fprs.append(fpr)
        model.cv_storage.tprs.append(tpr)
        model.cv_storage.model.append(clf)

    return model