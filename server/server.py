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
    
    return jsonify({"status": "received"}), 200

@app.route('/global_predict', methods=['GET'])
def global_ARIMA_predict():
    predictions, results, mae, plot_url = global_model.predict(
        file_path=r'E:\federated_stock_prediction\frontend\data\stock_data_node1.csv',
        # model_type='ARIMA',  # or 'SARIMAX'
        model_type='SARIMAX',  
        steps=30
    )
    if results is None:
        return jsonify({
            "error": "Model prediction failed. Please check the logs for details."
        }), 500
    results_list = results.reset_index().rename(columns={'index': 'Date'}).to_dict(orient='records')
    return jsonify({
        "predictions": predictions.tolist(),
        "results": results_list,
        "mae": mae,
        "plot_url": plot_url
    }), 200

if __name__ == "__main__":
    app.run(host='172.22.51.252', port=5000)
