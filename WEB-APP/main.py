from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load the LSTM model and scaler
model = load_model('models/LSTM-Stock-Market.keras')
scaler = joblib.load('models/stock-scaler.sav')

app = Flask(__name__)  # Initialize the Flask application

# Default values for the last 50 days
default_values = [
    60.6428, 60.6928, 60.9085, 61.1571, 61.9599, 62.0214, 60.9999, 60.2242, 60.0385, 57.8557,
    55.4242, 56.0914, 57.7128, 56.2199, 58.7464, 58.5442, 60.0642, 62.1571, 63.4942, 63.1114,
    64.4728, 65.1014, 66.4242, 65.5771, 65.6871, 65.4242, 64.5014, 64.8356, 62.7371, 60.4628,
    62.7214, 61.7014, 62.5928, 63.4357, 62.2785, 62.9785, 64.2714, 62.8571, 63.6642, 64.6428,
    64.3899, 64.7456, 63.6642, 63.6385, 62.3571, 63.5328, 62.2485, 62.7857, 61.7857, 62.1999
]

@app.route('/')
def index():
    return render_template('stock_details.html', default_values=default_values)

@app.route('/getprediction', methods=['POST'])
def getprediction():
    result = request.form

    # Collecting the stock prices for the last 50 days
    stock_prices = []
    for i in range(1, 51):
        day_price = float(result[f'day{i}'])
        stock_prices.append(day_price)

    # Convert the list to a numpy array and reshape for the model
    test_data = np.array(stock_prices).reshape(1, -1, 1)

    # Scale the input data
    test_data = scaler.transform(test_data.reshape(-1, 1)).reshape(1, -1, 1)

    # Make prediction
    prediction = model.predict(test_data)

    # Inverse scale the prediction
    prediction = scaler.inverse_transform(prediction)
    predicted_price = prediction[0][0]

    resultDict = {"name": result['name'], "symbol": result['symbol'], "predicted_price": round(predicted_price, 2)}

    return render_template('stock_results.html', results=resultDict)

if __name__ == '__main__':
    app.run(debug=True)
