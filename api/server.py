import os
import pandas as pd
from sklearn.externals import joblib
from flask import Flask, jsonify, request
import pickle
import sys 

path = "models/nayive_bayes.sav"

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def apicall():
    try:
        test_json = request.get_json()
        test = pd.DataFrame(test_json)
        print(test)

    except Exception as e:
        raise e

    clf = path

    print("Loading the model...")
    loaded_model = None
    with open('../'+clf, 'rb') as f:
        loaded_model = pickle.load(f)

    print("The model has been loaded...doing predictions now...", file=sys.stdout)
    predictions = loaded_model.predict(test['entities'])
    print(predictions, file=sys.stdout)
    prediction_series = list(pd.Series(predictions))

    final_predictions = pd.DataFrame(prediction_series)

    responses = jsonify(predictions=final_predictions.to_json(orient="records"))
    responses.status_code = 200

    return (responses)