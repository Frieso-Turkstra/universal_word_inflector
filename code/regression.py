from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, r_regression
from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import PredictionErrorDisplay
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# Read in the data.
features_df = pd.read_json("data/features.jsonl", lines=True)
scores_df = pd.read_json("data/scores_base.jsonl", lines=True)

# Shared task uses iso_codes, convert them to wals_codes.
iso2wals = {"amh": "amh", "arz": "aeg", "dan": "dsh", "eng": "eng", "fin": "fin", "fra": "fre", "grc": "","heb": "heb", "hun": "hun", "hye": "arm", "ita": "ita", "jap": "jpn", "kat": "geo", "mkd": "mcd", "rus": "rus", "spa": "spa", "swa": "swa", "tur": "tur", "nav": "nav", "afb": "arg", "sqi": "alb", "deu": "ger", "sme": "sno", "bel": "blr", "klr": "khg", "san": ""}
scores_df["wals_code"] = scores_df["iso_code"].apply(lambda x: iso2wals[x])
scores_df.drop(["iso_code"], axis=1, inplace=True)

# Combine.
df = pd.merge(features_df, scores_df, on=["wals_code"])

# One hot encode the morphological features.
features = pd.concat([pd.get_dummies(df[feature]) for feature in df.iloc[:, 10:-1].columns], axis=1)
target = df["accuracy"]

# Fit a linear regressor model using train_test_split.
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

regressor = LinearRegression(n_jobs=-1)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)

# Standardize the features by scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the MLPRegressor
mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', random_state=42)
mlp_regressor.fit(X_train_scaled, y_train)

# Predict accuracy scores on the test set
mlp_predictions = mlp_regressor.predict(X_test_scaled)

# And a linear regression using cross-validation.
cv_regressor = LinearRegression(n_jobs=-1)
cv = KFold(n_splits=5, random_state=0, shuffle=True)
cv_predictions = cross_val_predict(cv_regressor, features, target, cv=cv, n_jobs=-1)

# Visualize cross-validation predictions.
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
PredictionErrorDisplay.from_predictions(
    target,
    y_pred=cv_predictions,
    kind="actual_vs_predicted",
    subsample=100,
    ax=axs[0],
    random_state=0,
)
axs[0].set_title("Actual vs. Predicted values")
PredictionErrorDisplay.from_predictions(
    target,
    y_pred=cv_predictions,
    kind="residual_vs_predicted",
    subsample=100,
    ax=axs[1],
    random_state=0,
)
axs[1].set_title("Residuals vs. Predicted Values")
fig.suptitle("Plotting cross-validated predictions")
plt.tight_layout()
plt.show()

# Evaluate both linear regression models and MLP.
for metric in [mean_absolute_error, mean_squared_error]:
    print("train_test_split", metric.__name__, metric(y_test, predictions))
    print("cross-validation", metric.__name__, metric(target, cv_predictions))
    print("mlp", metric.__name__, metric(y_test, mlp_predictions))


# Print out the features from best to worst.
selector = SelectKBest(r_regression, k="all").fit(X_train, y_train)
top_scores = selector.scores_.argsort()
labels = [X_train.columns[i] for i in sorted(top_scores)]
coefficients = [regressor.coef_[i] for i in sorted(top_scores)]

df = pd.DataFrame(zip(labels, coefficients), columns=["feature", "coefficient"])
df = df.sort_values(by="coefficient", ascending=False)
print(df)

#wals2name = {"amh": "Armharic", "aeg": "Arabic (Egyptian)", "dsh": "Danish", "eng": "English", "fin": "Finnish", "fre": "French", "heb": "Hebrew (Modern)", "hun": "Hungarian", "arm": "Armenian (Eastern)", "ita": "Italian", "jpn": "Japanese", "geo": "Georgian", "mcd": "Macedonian", "rus": "Russian", "spa": "Spanish", "swa": "Swahili", "tur": "Turkish", "nav": "Navajo", "arg": "Arabic (Gulf)", "alb": "Albanian", "ger": "German", "sno": "Saami (Northern)", "blr": "Belorussian", "khg": "Khaling"}
