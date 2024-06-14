from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('car_price_prediction_Advanced_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        year = int(request.form['year'])
        present_price = float(request.form['present_price'])
        kms_driven = int(request.form['kms_driven'])
        fuel_type = int(request.form['fuel_type'])
        owner = int(request.form['owner'])
        seller_type = int(request.form['seller_type'])
        transmission_type = int(request.form['transmission_type'])

        # Prepare the new data for prediction
        new_data = pd.DataFrame({
            'Year': [year],
            'Present_Price': [present_price],
            'Kms_Driven': [kms_driven],
            'Fuel_Type': [fuel_type],
            'Owner': [owner],
            'Seller_Type_Individual': [seller_type],
            'Transmission_Manual': [transmission_type]
        })

        # Scale the new data using the loaded scaler
        new_data_scaled = scaler.transform(new_data)

        # Predict using the loaded model
        prediction = model.predict(new_data_scaled)

        # Return the prediction result
        return render_template('result.html', pred=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
