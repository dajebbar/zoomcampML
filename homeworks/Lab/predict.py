#!/usr/bin/env python
# coding: utf-8

import pickle
from flask import (
    Flask,
    request,
    jsonify
)


app = Flask(__name__)

with open('dv.bin', 'rb') as dv_in, open('model1.bin', 'rb') as model_in:
    dv = pickle.load(dv_in)
    model = pickle.load(model_in)

def score_client(client):
    X = dv.transform([client])
    return model.predict_proba(X)[0,1]

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    y_pred = score_client(client)
    
    card = y_pred >= .5
    
    return jsonify({
        'card_probability': float(y_pred.round(3)),
        'card': bool(card)
    })
    
if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    # client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
    # print(score_client(client))
