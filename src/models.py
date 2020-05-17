# Utility code
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.stats import randint as sp_randint
import utils
import pickle
import shap
import functools
from abc import ABC, abstractmethod

SEED = 0

class CVModelStorage:
    def __init__(self, auc=[], fprs=[], tprs=[], model=[]):
        self.auc = auc
        self.fprs = fprs
        self.tprs = tprs
        self.model = model
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.y_pred = []

    def get_mean_auc(self):
        return np.mean(self.auc)

class Model(ABC):
    def __init__(self, model_name, param_dict):
        """
        Arguments:
        model_name - name of model
        param_dict - params to be passed to model
        """

        #access actual self.model underneath for increased control if needed
        self.model_name = model_name
        self.model = self.init_model(model_name, param_dict)
        self.cv_storage = CVModelStorage([], [], [], []) #store results of each fold of cross val
        self.argmax_ind = -1 #stores the index of the model with best score, within cv_storage lists
        self.params_dict = param_dict
        self.train_splits = []

    @abstractmethod
    def init_model(self, model_name, param_dict):
        """ To be implemented by subclasses (linear and forest), big switch statement on model_name flag

            Returns model initialized with param_dict parameters
        """
        pass

    def avgd_folds_decision_function(self, test_set):
        temp = np.zeros(len(test_set))
        folds = len(self.cv_storage.model)
        for i in range(folds):
            k = pickle.loads(self.cv_storage.model[i])
            if hasattr(k, 'predict_proba'):
                temp += k.predict_proba(test_set)[:, 1]
            elif hasattr(k, 'decision_function'):
                temp += k.decision_function(test_set)
        return temp * (1 / folds)

    def avgd_folds_decision_function_multiclass(self, test_set):
        temp = []
        folds = len(self.cv_storage.model)
        print(folds)
        for i in range(folds):
            clf = pickle.loads(self.cv_storage.model[i])
            if hasattr(clf, 'predict_proba'):
                temp.append(clf.predict_proba(test_set))
            else:
                print("Multiclass currently only supports predict_proba")
        return np.dstack(temp).mean(axis=2)

    def decision_function(self, test_set):
        """ Consistency between predict_proba and decision_function
        """
        score = None
        if hasattr(self.model, 'predict_proba'):
            score = self.model.predict_proba(test_set)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            score = self.model.decision_function(test_set)
        return score

    def decision_function_multiclass(self, test_set):
        """ Return the multiclass predict_proba values
        """
        score = None
        if hasattr(self.model, 'predict_proba'):
            score = self.model.predict_proba(test_set)
        else:
            print("Multiclass currently only supports predict_proba")
        return score

    def run_cv(self, X, y, n_splits, seed=SEED):
        """ Run cross val with seed to standardize across models
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=seed)

        for train, test in cv.split(X, y):
            self.model.fit(X[train], y[train])
            score = self.decision_function(X[test])
            fpr, tpr, _ = roc_curve(y[test], score)
            a = auc(fpr, tpr)
            self.cv_storage.auc.append(a)
            self.cv_storage.X_train.append(X[train])
            self.cv_storage.y_train.append(y[train])
            self.cv_storage.X_test.append(X[test])
            self.cv_storage.y_test.append(y[test])
            self.cv_storage.y_pred.append(score)
            self.cv_storage.fprs.append(fpr)
            self.cv_storage.tprs.append(tpr)
            self.cv_storage.model.append(pickle.dumps(self.model))
            if isinstance(self, LinearModel):
                self.train_splits.append(X[train])

    def run_cv_multiclass(self, X, y, n_splits, seed=SEED):
        """ Run cross validation with seed to standardize across models
        Support for multiclass problems.
        """
        cv = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=seed)

        for train, test in cv.split(X, y):
            self.model.fit(X[train], y[train])
            score = self.decision_function_multiclass(X[test])

            # make a one hot multiclass label
            y_one_hot = np.zeros(score.shape)
            for idx, label in enumerate(y[test]):
                y_one_hot[idx, label] = 1

            fpr, tpr, _ = roc_curve(y_one_hot.ravel(), score.ravel())
            a = auc(fpr, tpr)

            self.cv_storage.auc.append(a)
            self.cv_storage.fprs.append(fpr)
            self.cv_storage.tprs.append(tpr)
            self.cv_storage.model.append(pickle.dumps(self.model))
            if isinstance(self, LinearModel):
                print(len(self.cv_storage.model))
                self.train_splits.append(X[train])

    def get_prediction_stats(self, true_values, pred_values):
        """ Gets the fpr, tpr, and AUC

        Arguments:
            true {List} -- True predictions
            pred {List} -- predict_proba output
        """
        fpr, tpr, _ = roc_curve(true_values, pred_values)
        a = auc(fpr, tpr)
        return fpr, tpr, a

    def update_model_parameters(self, params_dict):
        """ Update the model parameters to the params_dict
        Clears existing cv_storage
        """
        self.model = self.init_model(self.model_name, params_dict)
        self.params_dict = params_dict
        self.cv_storage = CVModelStorage([], [], [], [])

class LinearModel(Model):
    def init_model(self, model_name, param_dict):
        # Linear model subclass implementation
        self.train_splits = []
        self.is_elastic = False
        if model_name == 'logistic':
            return LogisticRegression(**param_dict)
        elif model_name == 'ridge':
            return RidgeClassifier(**param_dict)
        elif model_name == 'lasso':
            return LogisticRegression(**param_dict) #l1 penalty instead
        elif model_name == 'svm':
            return SVC(**param_dict)
        elif model_name == 'rbf':
            return SVC(**param_dict)
        elif model_name == 'elastic':
        	self.is_elastic = True
        	return LogisticRegression(**param_dict)
        return None

    def top_n_mz_coefficients(self, col_names, n=5):
        # Feature importance based on coefficients of linear model

        coefs = np.abs(self.model.coef_.ravel())
        ind = np.argpartition(coefs, -n)[-n:]
        ind = ind[np.argsort(coefs[ind])][::-1]
        return col_names[ind]

    def grid_search(self, nfolds, X, y, seed=SEED, visualize=True, label_sets=None):
        cv = StratifiedKFold(n_splits=nfolds, shuffle=False, random_state = seed)
        alphas = np.logspace(-1, 4, 20)
        tuned_params = [{'C': alphas}]
        if self.is_elastic:
        	l1_ratios = np.linspace(0, 1, num=20)
        	tuned_params = [{'C': alphas, 'l1_ratio' : l1_ratios}]
        if isinstance(self.model, RidgeClassifier):
            tuned_params = [{'alpha': alphas}]

        if label_sets:
            compute_auc = functools.partial(utils.compute_auc_binarized, label_sets=label_sets)
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False, scoring=compute_auc)
        else:
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False)

        grid.fit(X, y)

        if visualize:
            scores = grid.cv_results_['mean_test_score']
            scores_std = grid.cv_results_['std_test_score']
            plt.figure().set_size_inches(8, 6)
            plt.semilogx(alphas, scores)
            std_error = scores_std / np.sqrt(nfolds)

            # plot error lines showing +/- std. errors of the scores
            std_error = scores_std / np.sqrt(nfolds)

            plt.semilogx(alphas, scores + std_error, 'b--')
            plt.semilogx(alphas, scores - std_error, 'b--')

            # alpha=0.2 controls the translucency of the fill color
            plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

            plt.ylabel('CV score +/- std error')
            plt.xlabel('alpha')
            plt.axhline(np.max(scores), linestyle='--', color='.5')
            plt.xlim([alphas[0], alphas[-1]])

        return grid.best_params_

    def cv_shap_values(self, X_test, y_test):
        """ Feature importance with SHAP across all CV models

        Arguments:
            X_test {Arry} -- test samples
            y_test - useless, just for compatability with forests
        """
        models = [pickle.loads(i) for i in self.cv_storage.model]
        explainers = [shap.LinearExplainer(models[i], self.train_splits[i], feature_dependence="independent") for i in range(len(self.train_splits))]
        # explainers = [shap.KernelExplainer(models[i], self.train_splits[i], feature_dependence="independent") for i in range(len(self.train_splits))]
        # [shap.KernelExplainer(svm.predict_proba, X_train, link="logit")]
        total = []

        for explainer in explainers:
            shap_values = explainer.shap_values(X_test)
            if len(total) == 0:
                total = np.zeros(shap_values.shape)
            else:
                total += shap_values
        aggregated_shap = total / len(explainers)
        return aggregated_shap

    def cv_shap_values_multiclass(self, X_test, y_test):
        models = [pickle.loads(i) for i in self.cv_storage.model]
        explainers = [shap.LinearExplainer(models[i], self.train_splits[i], feature_dependence="independent") for i in range(len(self.train_splits))]
        shap_values = [explainer.shap_values(X_test) for explainer in explainers]
        num_classes = len(shap_values[0])
        output_shap = []
        for class_num in range(num_classes):
            class_shap_values = [shaps[class_num] for shaps in shap_values]
            stacked_shap_values = np.dstack(class_shap_values).mean(axis=2)
            output_shap.append(stacked_shap_values)

        return output_shap

class RandomForest(Model):
    def init_model(self, model_name, params_dict):
        clf = RandomForestClassifier(**params_dict)
        return clf

    def cv_shap_values(self, X_test, y_test):
        """ Feature importance with SHAP across all CV models

        Arguments:
            X_test {Array} -- Test Features
            y_test {Arry} -- Labels of test dataframe
        """
        models = [pickle.loads(i) for i in self.cv_storage.model]
        explainers = [shap.TreeExplainer(clf) for clf in models]
        shap_values = [explainer.shap_values(X_test, y_test) for explainer in explainers]
        aggregated_shap = np.dstack([shap_matrix[1] for shap_matrix in shap_values]).mean(axis=2)
        return aggregated_shap

    def cv_shap_values_multiclass(self, X_test, y_test):
        """ Feature importance with SHAP across all CV models.
        Support for multiclass problems.

        Arguments:
            X_test {Array} -- Test Features
            y_test {Arry} -- Labels of test dataframe

        Returns:
            output_shap {List} -- List of SHAP values for each class, ordered by class label.
        """
        models = [pickle.loads(i) for i in self.cv_storage.model]
        explainers = [shap.TreeExplainer(clf) for clf in models]
        shap_values = [explainer.shap_values(X_test, y_test) for explainer in explainers]
        num_classes = len(shap_values[0])
        output_shap = []
        for class_num in range(num_classes):
            class_shap_values = [shaps[class_num] for shaps in shap_values]
            stacked_shap_values = np.dstack(class_shap_values).mean(axis=2)
            output_shap.append(stacked_shap_values)

        return output_shap

    def random_search(self, nfolds, X, y, label_sets=None, seed=SEED):
        cv = StratifiedKFold(n_splits=nfolds, shuffle=False, random_state=seed)
        np.random.seed(seed)
        tuned_params = {
            'max_depth': [2, 4, 8, 16, 32],
            'n_estimators': [4, 16, 64, 256],
            'min_samples_split': [2, 4, 8, 16, 32]}

        if label_sets:
            compute_auc = functools.partial(utils.compute_auc_binarized, label_sets=label_sets)
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False, verbose=42, scoring=compute_auc)
            grid.fit(X, y)
        else:
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False, verbose=42)
            grid.fit(X, y)

        return grid.best_params_

class DecisionTree(RandomForest):
    def init_model(self, model_name, params_dict):
        clf = DecisionTreeClassifier(**params_dict)
        return clf

    def random_search(self, nfolds, X, y, label_sets=None, seed=SEED):
        cv = StratifiedKFold(n_splits=nfolds, shuffle=False, random_state=seed)
        np.random.seed(seed)
        tuned_params = {
            'max_depth': [1,2,3,4,5,7,8,16,32],
            'min_samples_split': [2,3,5,6,7,8,9,10,11,12,13,14,15,16,32]
        }

        if label_sets:
            compute_auc = functools.partial(utils.compute_auc_binarized, label_sets=label_sets)
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False, verbose=42, scoring=compute_auc)
            grid.fit(X, y)
        else:
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False, verbose=42)
            grid.fit(X, y)

        return grid.best_params_

class LGBM(Model):
    def init_model(self, model_name, params_dict):
        clf = lgb.LGBMClassifier(**params_dict)
        return clf

    def cv_shap_values(self, X_test, y_test):
        """ Feature importance with SHAP across all CV models

        Arguments:
            X_test {Array} -- Test Features
            y_test {Arry} -- Labels of test dataframe
        """
        models = [pickle.loads(i) for i in self.cv_storage.model]
        explainers = [shap.TreeExplainer(clf) for clf in models]
        shap_values = [explainer.shap_values(X_test, y_test) for explainer in explainers]
        aggregated_shap = np.dstack([shap_matrix[1] for shap_matrix in shap_values]).mean(axis=2)
        return aggregated_shap

    def cv_shap_values_multiclass(self, X_test, y_test):
        """ Feature importance with SHAP across all CV models.
        Support for multiclass problems.

        Arguments:
            X_test {Array} -- Test Features
            y_test {Arry} -- Labels of test dataframe

        Returns:
            output_shap {List} -- List of SHAP values for each class, ordered by class label.
        """
        models = [pickle.loads(i) for i in self.cv_storage.model]
        explainers = [shap.TreeExplainer(clf) for clf in models]
        shap_values = [explainer.shap_values(X_test, y_test) for explainer in explainers]
        num_classes = len(shap_values[0])
        output_shap = []
        for class_num in range(num_classes):
            class_shap_values = [shaps[class_num] for shaps in shap_values]
            stacked_shap_values = np.dstack(class_shap_values).mean(axis=2)
            output_shap.append(stacked_shap_values)

        return output_shap

    def random_search(self, nfolds, X, y, label_sets=None, seed=SEED):
        cv = StratifiedKFold(n_splits=nfolds, shuffle=False, random_state=seed)
        np.random.seed(seed)
        tuned_params = \
            {'num_leaves': [4,8,16,32,64,128],
             'max_depth': [2,4,8],
             'min_data_in_leaf': [2,4,8,16,32],
            }

        if label_sets:
            compute_auc = functools.partial(utils.compute_auc_binarized, label_sets=label_sets)
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False, verbose=42, scoring=compute_auc)
            grid.fit(X, y)
        else:
            grid = GridSearchCV(self.model, tuned_params, cv=cv, refit=False, verbose=42)
            grid.fit(X, y)

        return grid.best_params_
