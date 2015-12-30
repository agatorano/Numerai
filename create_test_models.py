from collections import Counter

import pandas as pd
import numpy as np

from scipy import stats

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import xgboost as xgb

import process_data


def create_test_model(model, x_train, y_train, x_val, y_val):
    '''
    train and run generic models
    '''

    model.fit(x_train, y_train)

    pred = model.predict(x_val)
    pred_auc = model.predict_proba(x_val)[:, 1]

    print(accuracy_score(y_val, pred))
    print(roc_auc_score(y_val, pred_auc))

    return model


def optimize_model(model, x_train, y_train, x_val, y_val, parameters):
    '''
    optimize model with grid_search
    '''

    clf = GridSearchCV(model, parameters)
    clf.fit(x_train, y_train)

    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))

    y_pred = clf.predict(x_val)
    print(classification_report(y_val, y_pred))

    return clf


def get_prediction_df(model_list, x_val, type='auc'):
    '''
    get a dataframe of predictions
    defaults to returning a df of probability scores
    each column is a model
    '''

    predictions = []
    for model in model_list:

        if type == 'auc':
            pred_auc = model.predict_proba(x_val.as_matrix())[:, 1]
            pred_auc = pd.Series(pred_auc)
            predictions.append(pred_auc)
        else:
            pred = model.predict(x_val.as_matrix())
            pred = pd.Series(pred)
            predictions.append(pred)

    predictions = pd.concat(predictions, axis=1)

    return predictions


def voting_ensemble_prediction(model_list, x_val, y_val):
    '''
    create a voting ensemble from a model list
    unweighted domacracy vote
    '''


    predictions = get_prediction_df(model_list, x_val, type='pred')

    results = []

    for i in range(len(predictions)):
        choice = Counter(predictions.iloc[i]).most_common(1)[0][0]
        results.append(choice)

    results = np.array(results)

    print(accuracy_score(y_val.as_matrix(), results))


def averaging_ensemble_prediction(model_list, x_val, y_val, ranks=None):
    '''
    averages probability scores of multiple models
    smooths out decision boundery of ensemble
    if ranks are sent average rank is determined
    '''

    if ranks is not None:
        ranks = ranks.mean(axis=1)
        pred_auc = preprocessing.normalize(ranks)[0]
    else:
        predictions = get_prediction_df(model_list, x_val)
        pred_auc = predictions.mean(axis=1)

    print(roc_auc_score(y_val.as_matrix(), pred_auc))

    return pred_auc


def rank_averaging_prediction(model_list, x_val, y_val):
    '''
    creates dataframe of probability score ranks
    sends the data frame to avaraging_ensemble_prediction
    for auc scoring
    '''

    predictions = get_prediction_df(model_list, x_val)

    ranks = []
    for col in predictions.columns:
        rank = pd.Series(stats.rankdata(predictions[col]))
        ranks.append(rank)

    ranks = pd.concat(ranks, axis=1)

    pred_auc = averaging_ensemble_prediction(model_list, x_val, y_val, ranks)

    return pred_auc


def main():
    df = process_data.read_data('numerai_datasets/numerai_training_data.csv')
    df = process_data.scale_data(df)
    df = process_data.get_c1_dummies(df)
    x_train, y_train, x_val, y_val = process_data.split_data(df)

    svc_model = create_test_model(SVC(probability=True, C=100, kernel='rbf'),
                                  x_train, y_train, x_val, y_val)
    process_data.save_model(svc_model, "models/svc/svc_model_c_100.pkl")


if __name__ == '__main__':
    main()
