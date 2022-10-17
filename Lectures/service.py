import bentoml
from bentoml.io import JSON


# model_ref = bentoml.xgboost.get("credit_risk_model:latest")
model_ref = bentoml.xgboost.get("credit_risk_model:bdgaflcobc24ftkl")

dv = model_ref.custom_objects["dico"]

# get access to the model
model_runner = model_ref.to_runner()

# create a service
svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def classify(application_data):
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)
    return {"Status": "Approved"}