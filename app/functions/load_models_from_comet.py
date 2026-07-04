from fastapi import APIRouter, HTTPException
import comet_ml
from os import getenv
from dotenv import load_dotenv
from comet_ml.integration.sklearn import load_model

predict_router = APIRouter()
load_dotenv()


def load_models_from_comet():
    """Charge les modèles depuis Comet ML"""
    try:
        api_key = getenv("COMET_API_KEY")
        workspace = getenv("COMET_WORKSPACE")
        project_name = getenv("COMET_PROJECT_NAME")

        if not all([api_key, workspace, project_name]):
            raise ValueError(
                "Les variables d'environnement COMET_API_KEY, COMET_WORKSPACE et COMET_PROJECT_NAME doivent être définies"
            )

        comet_ml.login(api_key=api_key)

        winner_model_path = f"registry://{workspace}/winner_model:1.3.0"
        score_model_path = f"registry://{workspace}/score_model:1.3.0"

        winner_model = load_model(winner_model_path)
        score_model = load_model(score_model_path)

        return winner_model, score_model

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du chargement des modèles depuis Comet ML: {str(e)}",
        )