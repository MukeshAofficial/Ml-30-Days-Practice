from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('car_price_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/car',methods=['POST'])
def predict():
    year=request.form['year']
    mileage=request.form['mileage']
    new_data = {
        'year': [int(year)],
        'mileage': [int(mileage)]
    }
    new_df = pd.DataFrame(new_data)
    
   
    prediction = model.predict(new_df)
    predicted_price = prediction[0]
    
    return render_template('result.html', pred=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)