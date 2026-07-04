from typing import List
from pydantic import BaseModel


class MatchPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    tournament: str


class ScorerPrediction(BaseModel):
    team: str
    scorer: str
    minute: int
    own_goal: bool
    penalty: bool


class MatchPredictionResponse(BaseModel):
    score_prediction: str
    scorers: List[ScorerPrediction]
