# pronotix-microservice

## Venv

```bash
python -m venv .venv
```

## Activate venv

```bash
# mac 
source .venv/bin/activate
# windows
.venv\scripts\activate
```

## Installation

```bash
pip install -r requirements.txt
```

## Freeze

```bash
pip freeze > requirements.txt
```

## Run with uvicorn

```bash
cd app && uvicorn main:app --reload --port 8002
```
## Initiate Comet-ml log-models

You will have to use the function **"save_and_log_models"** that you'll find in the .py file on path *app/functions/log_models_comet.py* , therefore add it after models creation like this :

```bash
model = create_models()
model.fit(X_train, y_predic) # or y_train in our case

save_and_log_models(model, metrics=metrics, model_names=model_names) # metrics and model_names are "none" by default, feel free to add your own when using this function, now after this, the model will be saved locally and on Comet
```# pronotix-microservice
