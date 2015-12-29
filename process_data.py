import numpy as np
import pandas as pd

from scipy import stats

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.covariance import EllipticEnvelope


def read_data(path):
    """
    read in the data
    """

    return pd.read_csv(path, header=0)


def scale_data(datframe):
    """
    scale data so that large numbers
    don't affect the algorithms
    """

    columns = datframe.columns[:14]
    np_dat = datframe[datframe.columns[:14]].as_matrix().astype(float)
    scaled = preprocessing.scale(np_dat)

    scaled = pd.concat([pd.DataFrame(data=scaled, columns=columns),
                        datframe[datframe.columns[14:]]], axis=1)

    return scaled


def get_c1_dummies(datframe):
    """
    Get dummies and rearange the columns
    so validation and target are last
    """

    dat_with_dummies = pd.get_dummies(datframe, prefix="",
                                      prefix_sep="", columns=['c1'])

    #rearange columns
    cols = dat_with_dummies.columns
    cols = cols[:14].append(cols[16:]).append(cols[14:16])
    dat_with_dummies = dat_with_dummies[cols]

    return dat_with_dummies


def split_data(datframe):
    """
    split data by the validation column
    into training and validation
    """

    val = datframe[datframe.validation == 1]
    train = datframe[datframe.validation == 0]

    val = val.drop('validation', axis=1)
    train = train.drop('validation', axis=1)

    y_train = train.target
    y_val = val.target

    x_train = train.drop('target', axis=1)
    x_val = val.drop('target', axis=1)

    return (x_train, y_train, x_val, y_val)


def pca_visualize(data):
    """
    visualize the data by PCA
    to check for possible outliers
    """
    reduced_data = PCA(n_components=2).fit_transform(data)
    xs, ys = reduced_data[:, 0], reduced_data[:, 1]

    df_show = pd.DataFrame(dict(x=xs, y=ys))

    # set up plot
    fig, ax = plt.subplots(figsize=(15, 13))  # set size
    ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

    ax.plot(df_show.x, df_show.y, marker='o', linestyle='',
            ms=12,  mec='none', alpha=0.1)
    ax.set_aspect('auto')
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(
        axis='y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')


    #plt.savefig("pca_2d_test.png")
    plt.show()


def outlier_detection(datframe, vis=0):
    """
    identify and remove outliers by EllipticalEnvelope
    visualize with PCA if desired
    """

    dat = datframe[datframe.columns[:14]]

    clf = EllipticEnvelope(contamination=.1)
    clf.fit(dat)
    y_pred = clf.decision_function(dat).ravel()

    outliers_fraction = 0.25
    threshold = stats.scoreatpercentile(y_pred, 100 * outliers_fraction)

    datframe['detect'] = y_pred
    datframe = datframe[datframe.detect > threshold]

    if vis == 1:
        pca_visualize(datframe[datframe.columns[:14]])

    return datframe


def save_model(model, name):
    '''
    saves trained model
    '''

    joblib.dump(model, name)


def load_model(name):
    '''
    loads pretrained model
    '''

    joblib.load(name)
