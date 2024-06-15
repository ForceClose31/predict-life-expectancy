import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Membaca data
df = pd.read_csv('data.csv')  # Ganti dengan nama file data Anda

# Memisahkan fitur (X) dan target (y)
X = df[['Age', 'Height', 'Weight', 'Gender']]
y = df['LifeExpectancy']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model RandomForestRegressor
model = RandomForestRegressor(random_state=42)

# GridSearch untuk penyetelan hyperparameter
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Model terbaik dari GridSearch
best_model = grid_search.best_estimator_

# Melakukan prediksi dengan model terbaik
y_pred = best_model.predict(X_test)

# Evaluasi model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
print(f'Best Parameters: {grid_search.best_params_}')
