import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("data.csv")  # encoding fix

# Select features
df = df[['bedrooms', 'bathrooms', 'floors', 'sqft_living', 'price']]

X = df[['bedrooms', 'bathrooms', 'floors', 'sqft_living']]
y = df['price']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling (Optional for Random Forest — tree based models don't need scaling)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Build Random Forest Model
rf_model = RandomForestRegressor(
    n_estimators=200,  # number of trees
    max_depth=None,    # grow until pure
    random_state=42,
    n_jobs=-1          # use all CPU cores
)

# Train Model
rf_model.fit(X_train, y_train)

# Predict
predictions = rf_model.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, predictions))
print("MAE:", mean_absolute_error(y_test, predictions))

# Save model
pickle.dump(rf_model, open("house_rf_model.pkl", "wb"))

print("Random Forest model trained & saved ✅")