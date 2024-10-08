
### 1. **`global_model.py`**

#### Purpose:

This file contains the `GlobalModel` class, which handles the aggregation of local models from different clients (nodes) and makes predictions based on the aggregated global model.

#### Code Explanation:

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class GlobalModel:
    def __init__(self):
        self.model_params = []  # List to store parameters of local models

    def aggregate_models(self, local_models):
        # Average the parameters of local models
        self.model_params = np.mean(local_models, axis=0)

    def predict(self, steps=30):
        # Use the aggregated parameters to make predictions
        model = ARIMA(endog=[], order=(self.model_params[0], self.model_params[1], self.model_params[2]))
        model_fit = model.fit()
        return model_fit.forecast(steps=steps)
```

- **`__init__`**: Initializes an empty list `model_params` to store parameters from each client’s local ARIMA model.
- **`aggregate_models`**: This function averages the parameters received from all local models to create a global model.
- **`predict`**: This function uses the aggregated parameters to create a global ARIMA model and forecast future stock prices for a specified number of steps.

### 2. **`server.py`**

#### Purpose:

This is the main server script that handles communication between the server and clients. It aggregates the models sent by clients and provides a prediction endpoint.

#### Code Explanation:

```python
from flask import Flask, request, jsonify
from global_model import GlobalModel

app = Flask(__name__)
global_model = GlobalModel()

@app.route('/send_model', methods=['POST'])
def receive_model():
    local_model = request.json['model_params']
    global_model.model_params.append(local_model)

    if len(global_model.model_params) == 3:  # Assuming 3 clients
        global_model.aggregate_models(global_model.model_params)
        global_model.model_params = []  # Reset after aggregation

    return jsonify({"status": "received"}), 200

@app.route('/global_predict', methods=['GET'])
def predict():
    predictions = global_model.predict(steps=30)
    return jsonify({"predictions": predictions.tolist()}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

- **`/send_model` endpoint**:

  - **Purpose**: Receives local model parameters from the clients and stores them in the `global_model.model_params` list.
  - **Process**:
    1. Accepts a POST request with the local model parameters.
    2. Appends the received parameters to the list.
    3. If all clients have sent their models (assuming 3 clients), it aggregates the models and resets the list.

- **`/global_predict` endpoint**:
  - **Purpose**: Allows clients or users to request predictions from the global model.
  - **Process**:
    1. Calls the `predict` method of the `GlobalModel` class to forecast the stock prices.
    2. Returns the predictions in JSON format.

### 3. **`utils.py`**

#### Purpose:

This file contains helper functions, mainly for data preprocessing.

#### Code Explanation:

```python
import pandas as pd

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df[['Date', 'Close']]  # Selecting only the Date and Close columns
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df['Close']
```

- **`preprocess_data` function**:
  - **Purpose**: Preprocesses the stock data before it is used to train the ARIMA model.
  - **Process**:
    1. Reads the CSV file containing the stock data.
    2. Filters out only the 'Date' and 'Close' columns.
    3. Converts the 'Date' column to a datetime object and sets it as the index.
    4. Returns the 'Close' prices as a time series.

### 4. **`client_nodeX.py` (e.g., `client_node1.py`)**

#### Purpose:

Each client script represents a different node in the federated learning setup. It trains a local ARIMA model on its specific dataset and sends the model parameters to the server.

#### Code Explanation:

```python
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
    local_model_params = train_local_model('data/stock_data_node1.csv')
    response = requests.post('http://server_ip:5000/send_model', json={"model_params": local_model_params.tolist()})
    print(response.json())
```

- **`train_local_model` function**:

  - **Purpose**: Trains a local ARIMA model on the provided stock data.
  - **Process**:
    1. Preprocesses the stock data using the `preprocess_data` function.
    2. Trains an ARIMA model with the order (5, 1, 0) on the preprocessed data.
    3. Returns the trained model parameters.

- **Main Script**:
  - **Purpose**: Runs the client script to train the local model and send the model parameters to the server.
  - **Process**:
    1. Calls the `train_local_model` function with a specific dataset.
    2. Sends a POST request to the server with the model parameters in JSON format.
    3. Prints the server's response.

### 5. **`frontend/app.py`**

#### Purpose:

This Flask application serves as the frontend for the federated learning system, allowing users to upload datasets and request predictions.

#### Code Explanation:

```python
from flask import Flask, render_template, request, redirect, url_for
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        file.save(f"data/{file.filename}")
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/predict', methods=['GET'])
def predict():
    response = requests.get('http://server_ip:5000/global_predict')
    predictions = response.json()['predictions']
    return render_template('predictions.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)
```

- **`index` route**:

  - **Purpose**: Displays the main page of the application.
  - **Template**: Renders `index.html`, which has links to upload data or request predictions.

- **`upload` route**:

  - **Purpose**: Handles file uploads for new stock data.
  - **Process**:
    1. If the request method is POST, saves the uploaded file to the `data/` directory.
    2. Redirects to the main page after successful upload.
  - **Template**: Renders `upload.html` for uploading files.

- **`predict` route**:
  - **Purpose**: Sends a GET request to the server to get the global model's predictions.
  - **Process**:
    1. Sends a request to the server to get predictions.
    2. Renders the `predictions.html` template with the prediction results.

### 6. **Frontend HTML Files**

- **`index.html`**: The main page with links to upload data and predict stock prices.
- **`upload.html`**: The upload page where users can upload new datasets.
- **`predictions.html`**: Displays the prediction results returned by the server.

### 7. **Frontend CSS (`style.css`)**

- Basic styling for the HTML pages to ensure a clean and user-friendly interface.

### Summary:

- **Global Model**: Aggregates local models from clients and predicts stock prices.
- **Server**: Coordinates communication between clients and the global model.
- **Clients**: Train local ARIMA models on their respective datasets and send the parameters to the server.
- **Frontend**: Provides a user-friendly interface to upload data and request predictions from the global model.

This setup simulates a federated learning environment where clients train models locally on their data, and the server aggregates these models to create a global model that can make more accurate predictions. The frontend allows easy interaction with the system.

# Data Splitting

Splitting the data into different parts is crucial for simulating a federated learning environment where each client (node) works with its own dataset. Here’s how you can split a dataset for use with multiple clients.

### Step-by-Step Guide to Splitting Data

Let's assume you have a dataset `google_stock_data.csv` that you want to split into three parts for three different clients.

### 1. **Load the Data**

First, load the data into a Pandas DataFrame.

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('google_stock_data.csv')

# Display the first few rows to understand the structure
print(data.head())
```

### 2. **Understand the Structure**

Typically, a stock dataset will have columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, etc. For our ARIMA model, we're primarily interested in the `Date` and `Close` columns.

### 3. **Shuffle the Data (Optional)**

If you want to ensure that each client gets a random sample of the data, you can shuffle it. However, for time series data, it might be more meaningful to split the data chronologically.

```python
# Optional: Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)
```

### 4. **Split the Data**

You can split the data into three equal parts or according to some other criteria. Here, we'll split it into three equal parts.

```python
# Calculate the split sizes
split_size = len(data) // 3

# Split the data into three parts
data_node1 = data.iloc[:split_size]
data_node2 = data.iloc[split_size:2*split_size]
data_node3 = data.iloc[2*split_size:]
```

### 5. **Save the Split Data**

Save each part as a separate CSV file to be used by each client.

```python
# Save the split data to separate CSV files
data_node1.to_csv('data/stock_data_node1.csv', index=False)
data_node2.to_csv('data/stock_data_node2.csv', index=False)
data_node3.to_csv('data/stock_data_node3.csv', index=False)
```

### Full Example Script

Here's a complete Python script to split your data:

```python
import pandas as pd

# Load the dataset
data = pd.read_csv('google_stock_data.csv')

# Display the first few rows to understand the structure
print("Original Data:")
print(data.head())

# Optional: Shuffle the data
# data = data.sample(frac=1).reset_index(drop=True)

# Calculate the split sizes
split_size = len(data) // 3

# Split the data into three parts
data_node1 = data.iloc[:split_size]
data_node2 = data.iloc[split_size:2*split_size]
data_node3 = data.iloc[2*split_size:]

# Save the split data to separate CSV files
data_node1.to_csv('data/stock_data_node1.csv', index=False)
data_node2.to_csv('data/stock_data_node2.csv', index=False)
data_node3.to_csv('data/stock_data_node3.csv', index=False)

print("Data split and saved successfully!")
```

### 6. **Verify the Split**

You can open the resulting CSV files to ensure they contain the expected data and are split correctly.

### 7. **Use the Split Data in Clients**

Each client script (`client_node1.py`, `client_node2.py`, etc.) will use its corresponding dataset:

- `client_node1.py` uses `data/stock_data_node1.csv`
- `client_node2.py` uses `data/stock_data_node2.csv`
- `client_node3.py` uses `data/stock_data_node3.csv`

### Summary

- Load the full dataset.
- Optionally shuffle the data.
- Split the data into parts (one for each client).
- Save the parts as separate CSV files.
- Use these CSV files in your client scripts.

This ensures that each client trains its model on a different subset of the data, simulating a federated learning scenario.
