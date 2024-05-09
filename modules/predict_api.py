import os
import glob
import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import warnings
warnings.filterwarnings('ignore')


path = os.environ.get('PROJECT_PATH', os.path.join(os.path.expanduser('~'), 'sber_auto'))

app = FastAPI()

model_file = glob.glob(f'{path}/data/models/*.pkl')

with open(*model_file, 'rb') as file:
    model = dill.load(file)
print(model['metadata'])

data = pd.read_csv(f'{path}/data/train/to_pipeline/df_train.csv')
class Form(BaseModel):
    client_id: float

    utm_source: str

    utm_medium: str

    utm_campaign: str

    utm_adcontent: str

    geo_country: str

    geo_city: str


class Prediction(BaseModel):
    client_id: float

    predict: int

@app.get('/status')
def status():
    return 'I am Fine'


@app.get('/version')
def version():
    return model['metadata']


@app.get('/read_mi')
def read_mi():
    to_request = 'model_taking: [client_id: float, utm_source: str, utm_medium: str, utm_campaign: str, utm_adcontent: str, geo_country: str, geo_city: str]'
    to_answer = 'answer_model: [0 - not: 1 - yes]'
    return to_request, to_answer


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)
    return {'id': form.client_id, 'predict': y[0]}


