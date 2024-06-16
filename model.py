import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib


# Function to categorize BMI into health status
def categorize_health_status(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 24.9:
        return 'Normal'
    elif bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'


# Load and preprocess data
df = pd.read_csv('data.csv')
df['Height_m'] = df['Height'] / 100
df['BMI'] = df['Weight'] / (df['Height_m'] ** 2)
df['HealthStatus'] = df['BMI'].apply(categorize_health_status)

# Separate features (X) and target (y) for life expectancy prediction
X_life = df[['Age', 'Height', 'Weight', 'Gender', 'BMI']]
y_life = df['LifeExpectancy']

# Split data into training and testing sets for life expectancy prediction
X_life_train, X_life_test, y_life_train, y_life_test = train_test_split(X_life, y_life, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor for life expectancy prediction
life_model = RandomForestRegressor(random_state=42)

# Define parameter grid for GridSearchCV
param_grid_life = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None]
}

# Perform GridSearchCV to find the best hyperparameters for life expectancy prediction
grid_search_life = GridSearchCV(estimator=life_model, param_grid=param_grid_life, cv=5, scoring='neg_mean_squared_error')
grid_search_life.fit(X_life_train, y_life_train)

# Best model from GridSearchCV for life expectancy prediction
best_life_model = grid_search_life.best_estimator_

# Save the best life expectancy model to a file
joblib.dump(best_life_model, 'best_life_model.pkl')

# Evaluate the life expectancy model
y_life_pred = best_life_model.predict(X_life_test)
mse = mean_squared_error(y_life_test, y_life_pred)
r2 = r2_score(y_life_test, y_life_pred)
print(f'Mean Squared Error (Life Expectancy Model): {mse}')
print(f'R-squared (Life Expectancy Model): {r2}')
print(f'Best Parameters for Life Expectancy Model: {grid_search_life.best_params_}')

# Example: Predict life expectancy for new data
new_data = pd.DataFrame({'Age': [35], 'Height': [170], 'Weight': [95], 'Gender': [1]})
new_data['Height_m'] = new_data['Height'] / 100
new_data['BMI'] = new_data['Weight'] / (new_data['Height_m'] ** 2)
predicted_life_expectancy = best_life_model.predict(new_data[['Age', 'Height', 'Weight', 'Gender', 'BMI']])

# Function to adjust life expectancy prediction based on BMI category
def adjust_life_expectancy(prediction, bmi):
    if bmi < 18.5:
        return prediction * 0.9  # Adjust for underweight
    elif bmi < 24.9:
        return prediction  # No adjustment for normal weight
    elif bmi < 29.9:
        return prediction * 0.95  # Adjust for overweight
    else:
        return prediction * 0.8  # Adjust for obesity

# Adjust predicted life expectancy based on BMI
adjusted_predicted_life_expectancy = adjust_life_expectancy(predicted_life_expectancy[0], new_data['BMI'].values[0])
print(f'Predicted Life Expectancy (Adjusted): {adjusted_predicted_life_expectancy}')
