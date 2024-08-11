import requests
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from utils import preprocess_data

def train_local_model(filepath):
    data = preprocess_data(filepath)
    model = ARIMA(data, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit.params

if __name__ == "__main__":
    local_model_params = train_local_model(r"E:\Data Splitting\data\stock_data_node1.csv")
    response = requests.post('http://172.22.51.252:5000/send_model', json={"model_params": local_model_params.tolist()})
    print(response.text)  # This will show the raw content of the response

