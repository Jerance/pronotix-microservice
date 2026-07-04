# pronotix-microservice

Pronotix Microservice is a FastAPI-based service that predicts football match outcomes using machine learning models trained on historical data. It can estimate:

- the likely winner of a match
- the predicted scoreline
- the most likely scorers
- a simple explanation of the prediction

## Project overview

This project uses a small machine learning pipeline built with Python, pandas, scikit-learn, and FastAPI. The service reads match data from the files in the app/data folder, prepares the features, trains models, and exposes them through HTTP endpoints.

The main components are:

- app/routes/predict.py: API endpoints for training and prediction
- app/functions/: data preparation, model creation, prediction logic, and Comet ML integration
- app/schemas/: request models used by the API
- app/data/: historical datasets used to train the models

## Environment setup

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the virtual environment

```bash
# macOS
source .venv/bin/activate

# Windows
.venv\scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Optional: update the requirements file

```bash
pip freeze > requirements.txt
```

## Run the service

From the project root, run:

```bash
uvicorn main:app --reload --port 8002
```

You can also start it with:

```bash
python main.py
```

## API endpoints

### POST /train

Trains the models from the historical datasets and logs them to Comet ML.

### POST /predict

Predicts the winner, scoreline, and likely scorers for a given match.

Example request body:

```json
{
  "home_team": "Paris Saint-Germain",
  "away_team": "Olympique de Marseille",
  "tournament": "Ligue 1"
}
```

### POST /explain

Returns a more detailed explanation of the prediction, including probability estimates and confidence level.

## Comet ML integration

To log and load models with Comet ML, define the following environment variables:

```bash
export COMET_API_KEY="your_api_key"
export COMET_WORKSPACE="your_workspace"
export COMET_PROJECT_NAME="your_project"
```

The training route uses the function save_and_log_models from app/functions/log_models_comet.py to save the trained models locally and on Comet.

## Notes

The project is intended as a simple demonstration of a machine learning microservice for football match prediction. It can be extended with better models, richer features, and more robust deployment configuration.
