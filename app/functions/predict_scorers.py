from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random


def train_scorer_model(goalscorers_data_path: str = "./data/goalscorers.csv"):
    df_scorers = pd.read_csv(goalscorers_data_path)

    # Encoder les équipes et les buteurs
    team_encoder = LabelEncoder()
    scorer_encoder = LabelEncoder()

    df_scorers["team_encoded"] = team_encoder.fit_transform(df_scorers["team"])
    df_scorers["scorer_encoded"] = scorer_encoder.fit_transform(df_scorers["scorer"])

    # Créer un modèle pour prédire les buteurs
    scorer_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Entraîner le modèle sur les données historiques
    X = df_scorers[["team_encoded"]]
    y = df_scorers["scorer_encoded"]

    scorer_model.fit(X, y)

    return scorer_model, team_encoder, scorer_encoder


def predict_match_scorers(
    home_team: str,
    away_team: str,
    predicted_home_score: int,
    predicted_away_score: int,
    goalscorers_data_path: str = "./data/goalscorers.csv",
) -> List[Tuple[str, str, int, bool, bool]]:
    # Charger et entraîner le modèle de prédiction des buteurs
    scorer_model, team_encoder, scorer_encoder = train_scorer_model(
        goalscorers_data_path
    )

    scorers_list = []

    try:
        # Prédire les buteurs probables pour l'équipe à domicile
        home_team_encoded = team_encoder.transform([home_team])
        home_probas = scorer_model.predict_proba(home_team_encoded.reshape(-1, 1))
        home_top_scorers_idx = np.argsort(home_probas[0])[-10:]
        home_scorers = scorer_encoder.inverse_transform(home_top_scorers_idx)

        # Prédire les buteurs probables pour l'équipe à l'extérieur
        away_team_encoded = team_encoder.transform([away_team])
        away_probas = scorer_model.predict_proba(away_team_encoded.reshape(-1, 1))
        away_top_scorers_idx = np.argsort(away_probas[0])[-10:]
        away_scorers = scorer_encoder.inverse_transform(away_top_scorers_idx)
    except (ValueError, KeyError):
        home_scorers = [f"Joueur {i+1} {home_team}" for i in range(3)]
        away_scorers = [f"Joueur {i+1} {away_team}" for i in range(3)]

    # Générer les minutes de but de manière aléatoire mais ordonnée
    all_minutes = sorted(
        random.sample(range(1, 90), predicted_home_score + predicted_away_score)
    )

    # Créer tous les buts possibles pour les deux équipes
    all_possible_goals = []

    # Buts de l'équipe à domicile
    for _ in range(predicted_home_score):
        scorer_idx = min(int(random.expovariate(0.5)), len(home_scorers) - 1)
        scorer = home_scorers[scorer_idx]
        own_goal = random.random() < 0.05  # 5% de chance de marquer contre son camp
        penalty = random.random() < 0.1  # 10% de chance que ce soit un penalty
        all_possible_goals.append((home_team, scorer, own_goal, penalty))

    # Buts de l'équipe à l'extérieur
    for _ in range(predicted_away_score):
        scorer_idx = min(int(random.expovariate(0.5)), len(away_scorers) - 1)
        scorer = away_scorers[scorer_idx]
        own_goal = random.random() < 0.05  # 5% de chance de marquer contre son camp
        penalty = random.random() < 0.1  # 10% de chance que ce soit un penalty
        all_possible_goals.append((away_team, scorer, own_goal, penalty))

    # Mélanger aléatoirement l'ordre des buts
    random.shuffle(all_possible_goals)

    # Associer les minutes de but avec les buteurs mélangés
    for (team, scorer, own_goal, penalty), minute in zip(
        all_possible_goals, all_minutes
    ):
        scorers_list.append((team, scorer, minute, own_goal, penalty))

    # Trier par minute
    return sorted(scorers_list, key=lambda x: x[2])
