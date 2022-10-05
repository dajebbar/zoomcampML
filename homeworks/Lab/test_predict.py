#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://0.0.0.0:9696/predict'

client_id = 'pca-145'
# client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}



response = requests.post(url, json=client).json()
print(response)
print()
if response['card']:
    print(f'Card for costumer with id: {client_id} has been Accepted!')
else:
    print(f'Card for costumer with id: {client_id} has been Rejected!')