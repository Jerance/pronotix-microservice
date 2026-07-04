from comet_ml import Experiment, login
from comet_ml.integration.sklearn import log_model
import cloudpickle
from os import getenv
from dotenv import load_dotenv

load_dotenv()

def init_comet_config():
    """
    Initialise la configuration Comet ML et retourne un objet Experiment
    """
    api_key = getenv("COMET_API_KEY")
    project_name = getenv("COMET_PROJECT_NAME")
    workspace = getenv("COMET_WORKSPACE")

    login(api_key=api_key)

    experiment = Experiment(
        api_key=api_key,
        project_name=project_name,
        workspace=workspace
    )

    return experiment

def save_and_log_models(experiment, models, metrics=None, model_names=None):
    """
    Sauvegarde et enregistre les modèles dans Comet ML Registry
    """
    if not model_names:
        model_names = [f"model_{i}" for i in range(len(models))]

    for model, name in zip(models, model_names):
        # Sauvegarder le modèle localement d'abord
        model_path = f"{name}.pkl"
        with open(model_path, "wb") as f:
            cloudpickle.dump(model, f)

        # Log le modèle dans l'expérience
        log_model(experiment, name, model, model_path, persistence_module=cloudpickle)

        # Enregistrer explicitement dans le registry
        experiment.log_model(name, model_path, overwrite=True)
        experiment.register_model(name)

    if metrics:
        experiment.log_metrics(metrics)

    experiment.end()
    return experiment