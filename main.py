from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the latest life expectancy model
best_life_model = joblib.load('best_life_model.pkl')

def categorize_health_status(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 24.9:
        return 'Normal'
    elif bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'

def get_health_recommendation(bmi_status):
    if bmi_status == 'Underweight':
        return 'You are underweight. It\'s important to focus on a balanced diet and consult a healthcare professional for guidance.'
    elif bmi_status == 'Normal':
        return 'Your weight is normal. Maintain a healthy lifestyle with regular exercise and a balanced diet.'
    elif bmi_status == 'Overweight':
        return 'You are overweight. Consider making dietary and lifestyle changes to achieve a healthier weight.'
    else:
        return 'You are obese. It\'s crucial to seek professional advice and make significant lifestyle changes for better health.'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    gender = int(request.form['gender'])

    height_m = height / 100
    bmi = weight / (height_m ** 2)

    bmi_status = categorize_health_status(bmi)

    health_recommendation = get_health_recommendation(bmi_status)

    # Predict life expectancy using the model
    predicted_life_expectancy = best_life_model.predict([[age, height, weight, gender, bmi]])[0]

    # Adjust life expectancy based on BMI category
    def adjust_life_expectancy(prediction, bmi_status):
        if bmi_status == 'Underweight':
            return prediction * 0.9  # Adjust for underweight
        elif bmi_status == 'Overweight':
            return prediction * 0.95  # Adjust for overweight
        elif bmi_status == 'Obesity':
            return prediction * 0.8  # Adjust for obesity
        else:
            return prediction  # No adjustment for normal weight

    adjusted_predicted_life_expectancy = adjust_life_expectancy(predicted_life_expectancy, bmi_status)

    return render_template('result.html', life_expectancy=adjusted_predicted_life_expectancy, bmi_status=bmi_status, health_recommendation=health_recommendation)


if __name__ == '__main__':
    app.run(debug=True)
