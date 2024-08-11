from flask import Flask, render_template, request, redirect, url_for
import requests
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if not os.path.exists('data'):
            os.makedirs('data')
        file.save(os.path.join('data', file.filename))
        return redirect(url_for('index'))
    return render_template('upload.html')

@app.route('/predict', methods=['GET'])
def predict():
    response = requests.get('http://172.22.51.252:5000/global_predict', timeout=100)
    print("Debugging response.json()-------\n")
    print(response.json())
    predictions = response.json()['predictions']
    results = response.json()['results']
    mae = response.json()['mae']
    plot_url = response.json()['plot_url']

    return render_template('predictions.html', predictions=predictions, results=results, mae=mae, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)
