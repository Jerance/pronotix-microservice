from fastapi import APIRouter, HTTPException
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.functions.prepare_data import prepare_data
from app.functions.create_models import create_models
from app.functions.predict_scorers import predict_match_scorers
from app.functions import init_comet_config, save_and_log_models
from app.functions.load_models_from_comet import load_models_from_comet

from app.schemas import MatchPredictionRequest

predict_router = APIRouter()


@predict_router.post("/train")
async def train_endpoint():
    """Endpoint pour entraîner et sauvegarder les modèles sur Comet ML"""
    try:
        df = pd.read_csv("./data/results.csv")
        df_prepared = prepare_data(df)

        X = df_prepared[["home_team", "away_team", "tournament"]]
        y_winner = df_prepared[["is_home_winner", "is_away_winner"]]
        y_score = df_prepared[["home_score", "away_score"]]

        X_train, X_test, y_winner_train, y_winner_test, y_score_train, y_score_test = (
            train_test_split(X, y_winner, y_score, test_size=0.2, random_state=42)
        )

        experiment = init_comet_config()

        winner_model, score_model = create_models()

        winner_model.fit(X_train, y_winner_train)
        score_model.fit(X_train, y_score_train)

        save_and_log_models(
            experiment=experiment,
            models=[winner_model, score_model],
            model_names=["winner_model", "score_model"],
        )

        return {
            "message": "Modèles entraînés et sauvegardés avec succès",
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'entraînement des modèles: {str(e)}",
        )


@predict_router.post("/predict")
async def predict_endpoint(match_data: MatchPredictionRequest):
    """Endpoint pour faire des prédictions en utilisant les modèles sauvegardés"""
    try:
        winner_model, score_model = load_models_from_comet()

        result = predict_match(
            winner_model,
            score_model,
            match_data.home_team,
            match_data.away_team,
            match_data.tournament,
        )

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}"
        )


def predict_match(
    winner_model: Pipeline,
    score_model: Pipeline,
    home_team: str,
    away_team: str,
    tournament: str,
):
    match_df = pd.DataFrame(
        {"home_team": [home_team], "away_team": [away_team], "tournament": [tournament]}
    )

    winner_pred = winner_model.predict(match_df)
    is_home_win = winner_pred[0][0]
    is_away_win = winner_pred[0][1]

    score_pred = score_model.predict(match_df)
    home_score = round(max(0, score_pred[0][0]))
    away_score = round(max(0, score_pred[0][1]))

    if not is_home_win and not is_away_win:
        if home_score != away_score:
            avg_score = round((home_score + away_score) / 2)
            home_score = away_score = avg_score

    scorers = predict_match_scorers(home_team, away_team, home_score, away_score)

    result = {
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "tournament": tournament,
        "scorers": [
            {
                "team": team,
                "scorer": scorer,
                "minute": minute,
                "own_goal": own_goal,
                "penalty": penalty,
            }
            for team, scorer, minute, own_goal, penalty in scorers
        ],
    }

    return result


@predict_router.post("/explain")
async def explain_prediction(match_data: MatchPredictionRequest):
    """Endpoint pour expliquer les prédictions du modèle"""
    try:
        winner_model, score_model = load_models_from_comet()
        
        # Créer le DataFrame pour la prédiction
        match_df = pd.DataFrame({
            "home_team": [match_data.home_team],
            "away_team": [match_data.away_team],
            "tournament": [match_data.tournament]
        })

        # Obtenir les probabilités du modèle winner
        winner_probas = winner_model.predict_proba(match_df)
        home_win_proba = winner_probas[0][0][0]  # Probabilité victoire domicile
        away_win_proba = winner_probas[0][0][1]  # Probabilité victoire extérieur
        draw_proba = 1 - (home_win_proba + away_win_proba)  # Probabilité match nul

        # Obtenir les prédictions de score
        score_pred = score_model.predict(match_df)
        home_score = round(max(0, score_pred[0][0]))
        away_score = round(max(0, score_pred[0][1]))

        # Calculer la confiance globale
        confidence = max(home_win_proba, away_win_proba, draw_proba)
        
        # Préparer l'explication
        explanation = {
            "match_context": {
                "home_team": match_data.home_team,
                "away_team": match_data.away_team,
                "tournament": match_data.tournament
            },
            "win_probabilities": {
                "home_win": round(float(home_win_proba) * 100, 2),
                "away_win": round(float(away_win_proba) * 100, 2),
                "draw": round(float(draw_proba) * 100, 2)
            },
            "score_prediction": {
                "predicted_home_score": home_score,
                "predicted_away_score": away_score,
                "score_explanation": _get_score_explanation(home_score, away_score)
            },
            "confidence_metrics": {
                "overall_confidence": round(float(confidence) * 100, 2),
                "prediction_strength": _get_confidence_level(confidence)
            },
            "interpretation": _generate_match_interpretation(
                match_data.home_team,
                match_data.away_team,
                home_win_proba,
                away_win_proba,
                home_score,
                away_score
            )
        }

        return explanation

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'explication de la prédiction: {str(e)}"
        )

def _get_confidence_level(confidence: float) -> str:
    """Convertit le niveau de confiance en description qualitative"""
    if confidence >= 0.8:
        return "Très forte"
    elif confidence >= 0.6:
        return "Forte"
    elif confidence >= 0.4:
        return "Modérée"
    else:
        return "Faible"

def _get_score_explanation(home_score: int, away_score: int) -> str:
    """Génère une explication du score prédit"""
    total_goals = home_score + away_score
    if total_goals == 0:
        return "Match à faible intensité offensive attendu"
    elif total_goals >= 4:
        return "Match à forte intensité offensive attendu"
    else:
        return "Match d'intensité offensive moyenne attendu"

def _generate_match_interpretation(
    home_team: str,
    away_team: str,
    home_win_proba: float,
    away_win_proba: float,
    home_score: int,
    away_score: int
) -> str:
    """Génère une interprétation narrative du match"""
    if home_win_proba > away_win_proba:
        favorite = home_team
        underdog = away_team
        win_proba = home_win_proba
    else:
        favorite = away_team
        underdog = home_team
        win_proba = away_win_proba

    if win_proba > 0.6:
        dominance = "nette domination"
    else:
        dominance = "légère domination"

    return (
        f"Le modèle prédit une {dominance} de {favorite} face à {underdog}. "
        f"Le score prévu de {home_score}-{away_score} reflète "
        f"{'une rencontre équilibrée' if abs(home_score - away_score) <= 1 else 'un écart significatif'} "
        f"entre les deux équipes."
    )
