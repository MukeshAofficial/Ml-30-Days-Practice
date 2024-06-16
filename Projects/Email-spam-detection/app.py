from flask import Flask, request, render_template
import joblib

# Load the trained model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('count_vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/spam', methods=['POST'])
def detect_spam():
    email_text = request.form['email']
    email_counts = vectorizer.transform([email_text])
    prediction = model.predict(email_counts)
    result = 'Spam' if prediction[0] == 1 else 'Not Spam'
    return render_template('result.html', pred=result)

if __name__ == '__main__':
    app.run(debug=True)
