# Utility code
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

SEED = 2019


class CustomPreprocessor():
    def __init__(self, transforms=['clip', 'power']):
        """Initialize the transformation attributes."""
        self.transforms = transforms
        self.feature_ceilings = None
        self.feature_floors = None
        self.transformer = None
        self.NOISE_SCALE = 0.0001
        self.CEILING_PERCENTILE = 99
        self.FLOOR_PERCENTILE = 100 - self.CEILING_PERCENTILE

    def fit(self, X_train):
        """Learns the transformation to use to preprocess the data."""
        for transformation in self.transforms:
            if transformation == 'power':
                self.transformer = preprocessing.PowerTransformer(method='yeo-johnson')
                self.transformer.fit(X_train)
            if transformation == 'standard':
                self.transformer = preprocessing.StandardScaler()
                self.transformer.fit(X_train)
            if transformation == 'quantile':
            	self.transformer = preprocessing.QuantileTransformer(output_distribution='normal')
            	self.transformer.fit(X_train)
            elif transformation == 'clip':
                self.feature_ceilings = np.percentile(X_train, self.CEILING_PERCENTILE, axis=0)
                self.feature_floors = np.percentile(X_train, self.FLOOR_PERCENTILE, axis=0)

    def transform(self, X):
        """Applies the learned transformation of the data"""
        X = X.copy()
        for transformation in self.transforms:
            if transformation == 'noise':
                X = X + self.NOISE_SCALE * np.random.random(X.shape)
            if transformation == 'power':
                X = self.transformer.transform(X)
            if transformation == 'quantile':
            	X = self.transformer.transform(X)
            elif transformation == 'clip':
                X = clip_features(X, self.feature_ceilings, self.feature_floors)
        return X


def clip_features(X, ceilings, floors):
    """Performs clipping in place.
    
    Args:
        X: the array to clip
        ceilings: the maximum value to set for each of the features.
        floors: the minimum value to set for each of the features.

    Returns:
        X: the modified array
    """
    X = X.copy()
    
    for i in range(X.shape[1]):
        X[X[:, i] > ceilings[i], i] = ceilings[i]
        X[X[:, i] < floors[i], i] = floors[i]
    return X


def load_data(feature_file_path, label_file_path, random_seed=SEED):
    """ Loads and processes data from the feature_file and label_file

    Arguments:
        feature_file_path {String} -- Path to the feature file
        label_file_path {String} -- Path to the label file
    """
    raw_df = pd.read_csv(feature_file_path, index_col=0)
    meta_df = pd.read_csv(label_file_path, index_col=0)
    df = raw_df.merge(meta_df.loc[:, ['virus_bins_option1', 'virus_bins_option2']], how='left', left_index=True, right_index=True)
    # sort to avoid any random components of merge
    df.sort_index()
    # shuffle
    df = df.sample(frac=1, random_state=random_seed)
    # rename cols
    df.rename(columns={'virus_bins_option1': 'flu', 'virus_bins_option2':'subtype'}, inplace=True)

    return df

def load_unformatted_data(feature_file_path, label_file_path, random_seed=SEED):
    """ Load and process data from the feature_file and label_file

    Arguments:
        feature_file_path {String} -- Path to the feature file
        label_file_path {String} -- Path to the label file

    Keyword Arguments:
        random_seed {Int} -- Ranom Seed (default: {SEED})
    """
    # load the labels as a dictionary
    labels = {}
    with open(label_file_path, 'r') as f:
        for l in f.readlines():
            name, label = l.strip().split(",")
            labels[name] = label

    # load the data as a pd.DataFrame
    df = pd.read_csv(feature_file_path, skiprows=2, index_col=0).transpose()
    label_col = [labels[i] for i in df.index.values]
    df['label'] = label_col

    return df

def map_label_column(df):
    """ For the expanded dataset. Map the label column to flu and subtype
    """
    df = df.loc[df.label != 'VTM', :].copy()
    mapping = {
        '2009 H1N1': 'Flu', 
        'Flu B': 'Flu', 
        'H3': 'Flu', 
        'negative':'negative',
    }
    flu = [mapping[i] for i in df.label.values]
    subtype = df.label.values
    df.drop(columns='label', inplace=True)
    df['flu'] = flu
    df['subtype'] = subtype
    
    return df

def encode_labels(df):
    """ Encode labels of loaded df as [0, n-1 classes]

    Arguments:
        df {pd.DataFrame} -- [description]

    Returns:
        df {pd.DataFrame} -- df with encoded labels
        flu_mapping {Dict} -- mapping for flu labels
        subtype_mapping {Dict} -- mapping for subtype labels
    """
    # Hardcode flu encoder because the only options are
    # Flu or negative
    flu_mapping = {'Flu': 1, 'negative': 0}
    df.flu = [flu_mapping[i] for i in df.flu.values]

    # more potential subtypes, use an encoder instead
    subtype_encoder = preprocessing.LabelEncoder()
    subtype_encoder.fit(df.subtype.values)
    df.subtype = subtype_encoder.transform(df.subtype.values)
    subtype_mapping = {i:idx for idx, i in enumerate(subtype_encoder.classes_)}

    return df, flu_mapping, subtype_mapping

def encode_viral_labels(df):
    """ Encode labels for viral dataset 
    
    Arguments:
        df {pd.DataFrame} 
    
    Returns:
        df {pd.DataFrame} -- df with encoded labels
        virus_mapping {Dict} -- coarse mapping for labels
        subtype_mapping 
    """
    # virus mapping
    virus_encoder = preprocessing.LabelEncoder()
    virus_encoder.fit(df.flu.values)
    df.flu = virus_encoder.transform(df.flu.values)
    virus_mapping = {i:idx for idx, i in enumerate(virus_encoder.classes_)}

    # more potential subtypes, use an encoder instead
    subtype_encoder = preprocessing.LabelEncoder()
    subtype_encoder.fit(df.subtype.values)
    df.subtype = subtype_encoder.transform(df.subtype.values)
    subtype_mapping = {i:idx for idx, i in enumerate(subtype_encoder.classes_)}

    return df, virus_mapping, subtype_mapping

def parse_tb():
    """ Parses TB dataset

    Returns:
        parsed_arr (numpy matrix) -- array with data, shape: (nsamples, nfeatures) (features are m/z values)
        labels - array with pos/neg classifications, shape: (nsamples,)
    """

    input_path = '2019_09_27_inputs_5000_cutoff.csv'
    output_path = '2019 09 27 outputs.csv'
    raw_df = pd.read_csv(input_path, skiprows=[0, 1], header=None)
    output = pd.read_csv(output_path, header=None)

    #find output labels using output df
    dic = {}
    i = 0
    for name in output[0]:
        label = output.iloc[i,1]
        i += 1
        if label == 'negative':
            dic[name] = 0
        elif label == 'positive':
            dic[name] = 1

    raw_arr = raw_df.to_numpy().T

    indices = []
    #drop all non pos/neg samples - can be changed later if we decide that the QC slots mean something
    i = 0
    for name in raw_arr[:,0]:
        if name in dic:
            indices.append(i)
        i += 1

    parsed_arr = raw_arr[indices]
    #encode labels
    labels = np.zeros(parsed_arr.shape[0])
    i = 0
    for name in parsed_arr[:,0]:
        labels[i] = dic[name]
        i += 1

    parsed_arr = parsed_arr[:,1:]

    return parsed_arr, labels

def train_test_split(df, test_size=30, random_seed=SEED):
    """ Randomly split df into train/test set dataframes.

    Arguments:
        df {pd.Dataframe} -- data df
        test_size {int} -- number of samples in the test set
    """
    df = df.sample(frac=1, random_state=random_seed)
    test_df = df.iloc[:test_size]
    train_df = df.iloc[test_size:]
    return train_df, test_df

def multiple_train_test_split(df, num_splits, subtype=True, random_seed=SEED):
    """ Randomly split df into train/test set dataframes
    Get the train_test split on 
    """
    output = []
    slf = StratifiedKFold(n_splits=num_splits, random_state=random_seed, shuffle=True)
    X, y = df_to_array(df, subtype=subtype)
    for train_idx, test_idx in slf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        output.append({
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_idx": train_idx,
            "test_idx": test_idx
        })
    return output

def df_to_array(df, subtype=False):
    """ Split the dataframe into X and y np arrays

    Arguments:
        df {pd.DataFrame} -- Pandas dataframe to split

    Keyword Arguments:
        subtype {bool} -- Use subtype labels instead of flu/not flu (default: {False})
    """
    if not subtype:
        X = df.iloc[:, :-2].values
        y = df.iloc[:, -2].values
    else:
        X = df.iloc[:, :-2].values
        y = df.iloc[:, -1].values

    return X, y

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

    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10,6))
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

def compute_auc(clf, X, y_true):
    try:
        y_pred = clf.predict_proba(X)
        y_one_hot = np.zeros(y_pred.shape)
        for idx, label in enumerate(y_true):
            y_one_hot[idx, label] = 1

        auc = roc_auc_score(y_one_hot, y_pred, average='micro')
    except:
        auc = .5

    return auc

def compute_auc_binarized(clf, X, y_true, label_sets):
    """ Compute a multiclass AUC for the subproblems in classification.
    Average the AUC across problems to get a single score.

    clf {sklearn Classifier} -- Classifier to use for evaluation
    X {np.array} -- features array
    y_true {np.array} -- labels array
    label_sets {List} -- List of lists of lists outlining the subproblems to evaluate
        For example: 
            [[[0,1], [2,3]], [[1], [0,2,3]]]
            This defines two label_sets: label (0,1) vs (2,3)
            and label (1) vs (0,2,3). The AUC for both is averaged.
    """
    y_pred = clf.predict_proba(X)
    aucs = []
    for label_set in label_sets:
        pos = y_pred[:, label_set[0]].sum(axis=1)
        pos_label = [1 if i in label_set[0] else 0 for i in y_true]
        auc = roc_auc_score(pos_label, pos)
        aucs.append(auc)
    auc = np.mean(aucs)

    return auc

def get_optimal_threshold(y_true, y_pred):
    thresholds = []
    sens_plus_spec = []
    for threshold in np.arange(0, 1, .01):
        preds = [1 if p > threshold else 0 for p in y_pred]
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)
        thresholds.append(threshold)
        sens_plus_spec.append(sensitivity + specificity)
    max_idx = np.argmax(sens_plus_spec)
    return thresholds[max_idx]