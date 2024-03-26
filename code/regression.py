from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, r_regression
from sklearn.linear_model import LinearRegression
import pandas as pd


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

# And using cross-validation.
cv_regressor = LinearRegression(n_jobs=-1)
cv = KFold(n_splits=5, random_state=0, shuffle=True)
cv_predictions = cross_val_predict(cv_regressor, features, target, cv=cv, n_jobs=-1)

# Evaluate both linear regression models.
for metric in [mean_absolute_error, mean_squared_error]:
    print("train_test_split", metric.__name__, metric(y_test, predictions))
    print("cross-validation", metric.__name__, metric(target, cv_predictions))

# Print out the features from best to worst.
selector = SelectKBest(r_regression, k="all").fit(X_train, y_train)
top_scores = selector.scores_.argsort()
labels = [X_train.columns[i] for i in sorted(top_scores)]
coefficients = [regressor.coef_[i] for i in sorted(top_scores)]

df = pd.DataFrame(zip(labels, coefficients), columns=["feature", "coefficient"])
df = df.sort_values(by="coefficient", ascending=False)
print(df)

#wals2name = {"amh": "Armharic", "aeg": "Arabic (Egyptian)", "dsh": "Danish", "eng": "English", "fin": "Finnish", "fre": "French", "heb": "Hebrew (Modern)", "hun": "Hungarian", "arm": "Armenian (Eastern)", "ita": "Italian", "jpn": "Japanese", "geo": "Georgian", "mcd": "Macedonian", "rus": "Russian", "spa": "Spanish", "swa": "Swahili", "tur": "Turkish", "nav": "Navajo", "arg": "Arabic (Gulf)", "alb": "Albanian", "ger": "German", "sno": "Saami (Northern)", "blr": "Belorussian", "khg": "Khaling"}
