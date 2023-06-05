from fastapi import FastAPI
from models import BankNoteRequest, BankNote

import pickle


app = FastAPI()

pickle_in = open(r'C:\Users\Sahil\DS ML tests\Project BankNote Authentication\classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

@app.get('/')
def index():
    return {'Welcome': 'Stranger'}


@app.post('/predict')
def predict_banknote(data: BankNoteRequest):
    data = BankNote(**data.dict())
    var = data.variance
    sk = data.skewness
    cu = data.curtosis
    ent = data.entropy
    #return {'variance':var}
    prediction = classifier.predict([[var, sk, cu, ent]])
    if prediction[0] > 0.5:
        res = 'Its a fake note'
    else:
        res = 'Its a banknote'
    return {'answer':res}