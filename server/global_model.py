import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
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
        # p, d, q = self.model_params
        self.model_params = np.abs(self.model_params)
        print("Aggregated model parameters:", self.model_params)


    def predict(self, file_path, steps=30, test_size=0.2):
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
            
            # Define ARIMA model
            model = ARIMA(endog=train, order=(5, 1, 0))
            
            try:
                # Attempt to fit the model without specifying start_params
                # model_fit = model.fit()  # Remove start_params for initial testing
                model_fit = model.fit(start_params=self.model_params)
                print("Model fitting successful", model_fit.params)
            except ValueError as e:
                print(f"ValueError during ARIMA model fitting: {e}")
                return None, None, None, None
            except np.linalg.LinAlgError as e:
                print(f"LinAlgError during ARIMA model fitting: {e}")
                return None, None, None, None
            except Exception as e:
                print(f"Error during ARIMA model fitting: {e}")
                return None, None, None, None
            
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
            results.index = results.index.strftime('%Y-%m-%d')  # Convert index to string for JSON serialization
            # print("Results DataFrame head:", results.head())
            # print("Results DataFrame info:", results.info())
            # print(results)
            
            # Plot the actual vs predicted values
            plt.figure(figsize=(10, 5))
            plt.plot(test.index, test, label='Actual')
            plt.plot(test.index, predictions, label='Predicted', linestyle='--')
            plt.title('ARIMA Model: Actual vs Predicted')
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


