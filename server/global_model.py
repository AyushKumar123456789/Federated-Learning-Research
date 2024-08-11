import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import io
import base64

class GlobalModel:
    def __init__(self):
        self.model_params = []  # List to store parameters of local models

    def aggregate_models(self, local_models):
        # Average the parameters of local models
        self.model_params = np.mean(local_models, axis=0)
        self.model_params = np.abs(self.model_params)
        print("Aggregated model parameters:", self.model_params)

    def _find_best_model(self, train):
        best_aic = np.inf
        best_order = None
        best_model = None

        # Define range for p, d, q
        p = d = q = range(0, 3)

        # Iterate over all combinations of p, d, q
        for i in p:
            for j in d:
                for k in q:
                    try:
                        # Fit ARIMA model
                        temp_model = ARIMA(endog=train, order=(i, j, k))
                        temp_model_fit = temp_model.fit()
                        if temp_model_fit.aic < best_aic:
                            best_aic = temp_model_fit.aic
                            best_order = (i, j, k)
                            best_model = temp_model_fit
                    except Exception as e:
                        continue
        print(f"Best ARIMA model order: {best_order} with AIC: {best_aic}")
        return best_model

    def _find_best_sarimax_model(self, train):
        best_aic = np.inf
        best_order = None
        best_model = None

        # Define range for p, d, q
        p = d = q = range(0, 3)

        # Iterate over all combinations of p, d, q
        for i in p:
            for j in d:
                for k in q:
                    try:
                        # Fit SARIMAX model
                        temp_model = SARIMAX(endog=train, order=(i, j, k), seasonal_order=(i, j, k, 12))
                        temp_model_fit = temp_model.fit(disp=False)
                        if temp_model_fit.aic < best_aic:
                            best_aic = temp_model_fit.aic
                            best_order = (i, j, k)
                            best_model = temp_model_fit
                    except Exception as e:
                        continue
        print(f"Best SARIMAX model order: {best_order} with AIC: {best_aic}")
        return best_model

    def predict(self, file_path, model_type='ARIMA', steps=30, test_size=0.05):
        try:
            # Read and preprocess the data
            data = pd.read_csv(file_path)
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            data.sort_index(inplace=True)
            if not data.index.freq:
                data = data.asfreq('D')

            time_series_data = data['close']
            if time_series_data.isna().sum() > 0:
                time_series_data = time_series_data.fillna(method='ffill').replace([np.inf, -np.inf], np.nan)
            
            train_size = int(len(time_series_data) * (1 - test_size))
            train, test = time_series_data[:train_size], time_series_data[train_size:]
            
            print("Training size:", len(train))
            print("Testing size:", len(test))
            
            # Select and fit the best model based on model_type
            if model_type == 'ARIMA':
                model_fit = self._find_best_model(train)
            elif model_type == 'SARIMAX':
                model_fit = self._find_best_sarimax_model(train)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Forecast the test period
            predictions = model_fit.forecast(steps=len(test))
            
            # Check if predictions are valid
            if len(predictions) != len(test):
                raise ValueError("Mismatch in length between predictions and test data.")
            
            # Calculate error
            mae = mean_absolute_error(test, predictions)
            print(f"Mean Absolute Error: {mae}")
            
            # Tabulate the actual vs predicted values
            results = pd.DataFrame({
                'Actual': test.values,
                'Predicted': predictions
            })
            results.index = test.index.strftime('%Y-%m-%d')
            
            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 5))
            plt.plot(test.index, test, label='Actual')
            plt.plot(test.index, predictions, label='Predicted', linestyle='--')
            plt.title(f'{model_type} Model: Actual vs Predicted')
            plt.xlabel('Date')
            plt.ylabel('Close Price')
            plt.legend()

            # Convert plot to PNG image and encode to base64
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close()  # Close the plot to free up memory

            return predictions, results, mae, plot_url
        
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, None, None, None
