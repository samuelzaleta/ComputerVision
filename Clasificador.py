import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from AttributerRemover import AttributerRemover


def main():
    dataset = pd.read_csv('dataset.csv', thousands=',')
    y = np.c_[dataset['Posicion']].ravel()
    X = np.c_[dataset.drop(labels='Posicion', axis=1)]
    # Scaling the params
    X = X / 360

    '''
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
    for train_index, test_index in split.split(X, y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
    '''

    '''
    possibleModels = [
        # GridSearchCV(SGDClassifier(random_state=42), [{
        #    'loss': ['hinge', 'log', 'perceptron'],
        #    'penalty': ['l2', 'l1', 'elasticnet'],
        #    'alpha': [0.0001, 0.001, 0.00001]
        # }]),
        # GridSearchCV(RandomForestClassifier(random_state=42), [{
        #    'criterion': ['gini', 'entropy'],
        #    'max_depth': [1, 2, 3, 5, 7, 9, 11],
        #    'n_estimators': [20, 30, 40, 50, 60, 70, 80, 90, 100],
        # }]),
        #
        # GridSearchCV(SVC(random_state=42), [
        #    {
        #        'kernel': ['linear'],
        #        'C': [100000, 10000.0, 1000.0, 100.0],
        #    },
        #    {
        #        'kernel': ['rbf', 'sigmoid'],
        #        'gamma': ['scale', 'auto'],
        #        'C': [100000, 10000.0, 1000.0, 100.0],
        #    },
        #    {
        #        'kernel': ['poly'],
        #        'degree': [1, 2, 3, 5, 7],
        #        'gamma': ['scale', 'auto'],
        #        'C': [100000, 10000.0, 1000.0, 100.0],
        #    }
        # ])
    ]
    '''

    pipeline = Pipeline([
        ('attributes', AttributerRemover(remove_hombro_hombro_der=True)),
    ])

    X = pipeline.transform(X)
    model = SVC(random_state=42, C=100)
    model.fit(X, y)

    dump(model, 'model.sav')

    '''
    model = GridSearchCV(
        pipeline, {
            'attributes__remove_hombro_hombro_der': [True, False],
            'attributes__remove_cadera_hombro_der': [True, False],
            'attributes__remove_hombro_hombro_izq': [True, False],
            'attributes__remove_cadera_hombro_izq': [True, False],
            'model__C': [100, 1000, 10000, 100000],
        },
    )
    model.fit(X_train, y_train)

    print(model.best_estimator_)
    print(model.best_params_)

    y_train_score = cross_val_score(model, X_train, y_train, cv=3)
    y_train_predic = cross_val_predict(model, X_train, y_train, cv=3)

    print(y_train_score)
    print(confusion_matrix(y_train, y_train_predic))
    '''

    '''
    for train_index, test_index in skfolds.split(X, y):
        clone_model = clone(model)
        X_train_folds = X[train_index]
        y_train_folds = y[train_index]
        X_test_folds = X[test_index]
        y_test_folds = y[test_index]

        clone_model.fit(X_train_folds, y_train_folds)
        y_pred = clone_model.predict(X_test_folds)
        n_correct = sum(y_pred == y_test_folds)
        print(n_correct / len(y_pred))
    '''


if __name__ == '__main__':
    main()
