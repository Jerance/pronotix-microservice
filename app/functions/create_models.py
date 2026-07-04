from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def create_models():
    """Crée et configure les modèles de prédiction"""
    categorical_features = ["home_team", "away_team", "tournament"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    # Modèle pour prédire le gagnant
    base_classifier = LogisticRegression(
        penalty="l2", max_iter=500, n_jobs=-1, random_state=42
    )
    winner_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", MultiOutputClassifier(base_classifier, n_jobs=-1)),
        ]
    )

    # Modèle pour prédire les scores
    base_regressor = RandomForestRegressor(
        n_estimators=50, max_depth=10, n_jobs=-1, random_state=42
    )
    score_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", MultiOutputRegressor(base_regressor, n_jobs=-1)),
        ]
    )

    return winner_model, score_model
