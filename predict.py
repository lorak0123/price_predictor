import pandas as pd
import numpy as np
from sklearn.svm import SVR


def load_data(namefile: str) -> pd.DataFrame:
    return pd.read_csv(namefile)


def prepare_data(df: pd.DataFrame, prediction_days: int) -> pd.DataFrame:
    df.reset_index(inplace=True, drop=True)
    df = pd.DataFrame(df['Close'])
    df['Prediction'] = df[['Close']].shift(-prediction_days)
    return df


def get_X_and_y(df: pd.DataFrame, prediction_days: int) -> (np.array, np.array):
    X = np.array(df.drop(['Prediction'], axis=1))
    X = X[:len(df) - prediction_days]

    y = np.array(df['Prediction'])
    y = y[:-prediction_days]
    return X, y


def get_X_predict(df: pd.DataFrame, prediction_days: int) -> np.array:
    X_predict = np.array(df.drop(['Prediction'], axis=1))[-prediction_days:]
    return X_predict


def get_fitted_model(X: np.array, y: np.array) -> SVR:
    model = SVR(kernel='rbf', C=1e3, gamma=0.00001)  # Create the model
    model.fit(X, y)  # Train the model
    return model


def get_predict(X_predict: np.array, model: SVR) -> np.array:
    predict = model.predict(X_predict)
    return predict


if __name__ == "__main__":
    prediction_days = 30
    df = prepare_data(load_data('test2.csv'), prediction_days)
    X, y = get_X_and_y(df, prediction_days)
    X_predict = get_X_predict(df, prediction_days)
    model = get_fitted_model(X, y)
    predict = get_predict(X_predict, model)
    print(predict)
