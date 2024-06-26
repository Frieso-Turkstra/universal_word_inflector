"""
This program tries to predict the accuracy scores of a language on the task
of morphological (re-)inflection based on its morphological features.
In total, three models generate predictions:
1) a linear regression model
2) a linear regression model with cross-validation
3) a multi-layer perceptron
All models are evaluated on Mean Absolute Error and Mean Squared Error.
Additionally, visualisations can be generated.
"""

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, r_regression
from sklearn.metrics import PredictionErrorDisplay
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
import pandas as pd
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--accuracy_file_path", "-a",
                        required=True,
                        help="Path to the file with accuracy scores.",
                        type=str,
                        )
    parser.add_argument("--features_file_path", "-f",
                        required=True,
                        help="Path to the file with features.",
                        type=str,
                        )
    parser.add_argument("--plot", "-p",
                        required=False,
                        help="Whether to generate a plot for the cross-validation predictions.",
                        action="store_true"
                        )
    
    args = parser.parse_args()
    return args


def plot_cv_predictions(predictions):
    # Visualize cross-validation predictions.
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    PredictionErrorDisplay.from_predictions(
        target,
        y_pred=predictions,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
    )
    axs[0].set_title("Actual vs. Predicted values")
    PredictionErrorDisplay.from_predictions(
        target,
        y_pred=predictions,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    plt.show()


def get_best_features(X_train, y_train, regressor):
    # Print out the features from best to worst.
    selector = SelectKBest(r_regression, k="all").fit(X_train, y_train)
    top_scores = selector.scores_.argsort()
    labels = [X_train.columns[i] for i in sorted(top_scores)]
    coefficients = [regressor.coef_[i] for i in sorted(top_scores)]

    df = pd.DataFrame(zip(labels, coefficients), columns=["feature", "coefficient"])
    df = df.sort_values(by="coefficient", ascending=False)
    return df
   

if __name__ == "__main__":

    # Read in the data and command-line arguments.
    args = create_arg_parser()
    features_df = pd.read_json(args.features_file_path, lines=True)
    scores_df = pd.read_json(args.accuracy_file_path, lines=True)

    # Shared task uses iso_codes, convert them to wals_codes.
    iso2wals = {
        "amh": "amh", "arz": "aeg", "dan": "dsh", "eng": "eng", "fin": "fin", "fra": "fre",
        "grc": "","heb": "heb", "hun": "hun", "hye": "arm", "ita": "ita", "jap": "jpn",
        "kat": "geo", "mkd": "mcd", "rus": "rus", "spa": "spa", "swa": "swa", "tur": "tur",
        "nav": "nav", "afb": "arg", "sqi": "alb", "deu": "ger", "sme": "sno", "bel": "blr",
        "klr": "khg", "san": ""
        }
    scores_df["wals_code"] = scores_df["iso_code"].apply(lambda x: iso2wals[x])
    scores_df.drop(["iso_code"], axis=1, inplace=True)

    # Combine.
    df = pd.merge(features_df, scores_df, on=["wals_code"])

    # One hot encode the morphological features.
    features = pd.concat([pd.get_dummies(df[feature]) for feature in df.iloc[:, 10:-2].columns], axis=1)
    target = df["accuracy"]

    # Split the data in a train and test set.
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Fit and predict with linear regressor model.
    regressor = LinearRegression(n_jobs=-1)
    regressor.fit(X_train, y_train)
    predictions_lr = regressor.predict(X_test)

    # Fit and predict with linear regression and cross-validation.
    regressor_cv = LinearRegression(n_jobs=-1)
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    predictions_cv = cross_val_predict(regressor_cv, features, target, cv=cv, n_jobs=-1)
    
    # Standardize the features by scaling.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the MLPregressor.
    regressor_mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
    regressor_mlp.fit(X_train_scaled, y_train)

    # Predict accuracy scores on the test set.
    predictions_mlp = regressor.predict(X_test_scaled)

    # Evaluate both linear regression models and MLP.
    for metric in [mean_absolute_error, mean_squared_error]:
        print("train_test_split", metric.__name__, metric(y_test, predictions_lr))
        print("cross-validation", metric.__name__, metric(target, predictions_cv))
        print("multi_layer_perceptron", metric.__name__, metric(y_test, predictions_mlp))

    # Print and save the best features.
    best_features_df = get_best_features(X_train, y_train, regressor)
    best_features_df.to_json("best_features.jsonl", lines=True, orient="records")
    print(best_features_df)

    # Plot cross-validation predictions if enabled.
    if args.plot:
        plot_cv_predictions(predictions_cv)
