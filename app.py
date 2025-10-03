from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("C:/Users/khush/OneDrive/Desktop/Mini-Project/Sonar_Model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form['features']

    numbers = [float(x) for x in input_data.split(',')]
    arr = np.array(numbers).reshape(1,-1)

    prediction = model.predict(arr)[0]

    result = 'Rock' if prediction == 'R' else 'Mine'

    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)