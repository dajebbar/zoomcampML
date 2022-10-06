import pickle


with open('dv.bin', 'rb') as dv_in, open('model1.bin', 'rb') as model_in:
    dv = pickle.load(dv_in)
    model = pickle.load(model_in)

def score_client(client):
    X = dv.transform([client])
    return model.predict_proba(X)[0, 1]


if __name__=='__main__':
    client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}
    print(score_client(client))