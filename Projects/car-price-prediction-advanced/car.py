import joblib
import pandas as pd

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('car_price_prediction_Advanced_model.pkl')

# Function to predict car price
def predict_car_price(year, present_price, kms_driven, fuel_type, owner, seller_type, transmission_type):
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

    return prediction[0]

# Example usage
year = 2015
present_price = 5.0
kms_driven = 30000
fuel_type = 0  # Petrol
owner = 0
seller_type = 0  # Dealer
transmission_type = 0  # Manual

predicted_price = predict_car_price(year, present_price, kms_driven, fuel_type, owner, seller_type, transmission_type)
print(f"The predicted selling price is: ₹{predicted_price:.2f} lakhs")
