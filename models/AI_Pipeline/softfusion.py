import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# load data
data = pd.read_csv('data.csv')

X = data[['mq135']]
y = data['co2']

# train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# save model
joblib.dump(model, 'model.pkl')

print("Model trained!")
