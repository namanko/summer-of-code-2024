from fastapi import FastAPI
import uvicorn
import pickle
from pydantic import BaseModel


with open('fraud_detection_model_fin.pkl','rb') as myf:
    model = pickle.load(myf)

# Creating FastAPI instance
app = FastAPI()


# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    amount : float
    type_CASH_IN : int
    type_CASH_OUT : int
    type_DEBIT : int
    type_PAYMENT : int
    type_TRANSFER : int
    Origdiff : float
    Destdiff : float



@app.get('/')
def index():
    return {'message': 'Fraud Detection ML API'}
# Creating an Endpoint to receive the data
# to make prediction on.
@app.post('/predict')
def predict(data: request_body):
    # Making the data in a form suitable for prediction
    data = data.dict()
    test_data = [[
        data['amount'],
        data['type_CASH_IN'],
        data['type_CASH_OUT'],
        data['type_PAYMENT'],
        data['type_TRANSFER'],
        data['type_DEBIT'],
        data['Origdiff'],
        data['Destdiff']
    ]]
    # Predicting the Class
    res = model.predict(test_data)[0]
    # Return the Result
    return {'isFraud': int(res)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
