import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# Load the datasets
weather_data = pd.read_csv('./datasets/weather_data.csv')
air_quality_data = pd.read_excel('./datasets/AirQualityUCI.xlsx')

# Preprocess weather data
weather_data.dropna(inplace=True)
X_weather = weather_data[['Temperature_C', 'Humidity_pct', 'Precipitation_mm', 'Wind_Speed_kmh']]
y_weather = pd.cut(weather_data['Temperature_C'], bins=3, labels=['Low', 'Medium', 'High'])

# Split data for classification
X_weather_train, X_weather_test, y_weather_train, y_weather_test = train_test_split(
    X_weather, y_weather, test_size=0.2, random_state=42)

# Define and train the classification model
classifier = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X_weather_train.columns)
        ])),
    ('classifier', RandomForestClassifier(n_estimators=100))
])
classifier.fit(X_weather_train, y_weather_train)

# Preprocess air quality data
air_quality_data['Date'] = air_quality_data['Date'].astype(str)
air_quality_data['Time'] = air_quality_data['Time'].astype(str)

# Combine 'Date' and 'Time' into 'DateTime'
air_quality_data['DateTime'] = pd.to_datetime(air_quality_data['Date'] + ' ' + air_quality_data['Time'])

# Set 'DateTime' as index and drop original 'Date' and 'Time' columns
air_quality_data.set_index('DateTime', inplace=True)
air_quality_data.drop(['Date', 'Time'], axis=1, inplace=True)
air_quality_data.fillna(method='ffill', inplace=True)

# Example: Predicting CO(GT) as a regression target
X_air_quality = air_quality_data.drop('CO(GT)', axis=1)
y_air_quality = air_quality_data['CO(GT)']

# Split data for regression
X_air_quality_train, X_air_quality_test, y_air_quality_train, y_air_quality_test = train_test_split(
    X_air_quality, y_air_quality, test_size=0.2, random_state=42)

# Define and train the regression model
regressor = Pipeline(steps=[
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X_air_quality_train.columns)
        ])),
    ('regressor', RandomForestRegressor(n_estimators=100))
])
regressor.fit(X_air_quality_train, y_air_quality_train)

# Save models
joblib.dump(regressor, 'air_quality_model.pkl')
joblib.dump(classifier, 'weather_model.pkl')

print("Models have been trained and saved successfully.")
