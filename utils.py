from typing import Tuple, Union, List
import numpy as np
from sklearn.linear_model import LogisticRegression
import openml

XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LogRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LogisticRegression) -> LogRegParams:
    """Returns the paramters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
    model: LogisticRegression, params: LogRegParams
) -> LogisticRegression:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LogisticRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    n_classes = 4  # label has 4 levels
    n_features = 128  # Number of features in dataset
    model.classes_ = np.array([i for i in range(10)])

    model.coef_ = np.zeros((n_classes, n_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_classes,))

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def load_data():
    num_cols = [
                "number of rooms", 
                "security level of the community",
                "residence space",
                # "building space",
                "noise level",
                "waterfront",
                "view",
                "air quality level",
                "aboveground space ",
                # "basement space",
                "building year",
                # "decoration year",
                # "lat",
                # "lng",
                # "total cost"
                ]
    
    cat_cols = [
        "city",
        "zip code",
    ]
    feature_names = cat_cols + num_cols
    df = pd.read_csv("./Train_Data_For_Task2.csv")
    features = df[feature_names]
    features = pd.get_dummies(features, columns=cat_cols)
    
    scaler = StandardScaler()
    features[num_cols] = scaler.fit_transform(features[num_cols])

    features = features.to_numpy()
    labels = df["label"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=666)
    return (x_train, y_train), (x_test, y_test)



def shuffle(X: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle X and y."""
    rng = np.random.default_rng()
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def partition(X: np.ndarray, y: np.ndarray, num_partitions: int) -> XYList:
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )


if __name__ == "__main__":
    load_data()