import pandas as pd

from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

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
