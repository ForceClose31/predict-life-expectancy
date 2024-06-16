from flask import Flask, request, render_template
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('life_expectancy_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    height = int(request.form['height'])
    weight = int(request.form['weight'])
    gender = int(request.form['gender'])
    
    input_data = pd.DataFrame([[age, height, weight, gender]], columns=['Age', 'Height', 'Weight', 'Gender'])
    prediction = model.predict(input_data)

    return render_template('index.html', prediction_text=f'Prediksi Harapan Hidup: {prediction[0]:.2f} tahun')

if __name__ == "__main__":
    app.run(debug=True)
