import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


np.random.seed(0) #pratek choti same seed

n= 400 #samples

study_hours = np.random.normal(5,2,n) 
study_hours = np.clip(study_hours, 0, None) # Ensure no negative study houre
sleep_hours = np.random.normal(7,1,n)
sleep_hours = np.clip(sleep_hours, 0, None) # Ensure no negative sleep hours
attendance = np.random.normal(80,10,n)
attendance = np.clip(attendance, 0, 100) # Ensure attendance is between 0 and 100

df = pd.DataFrame({
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'attendance': attendance
})

X = df[['study_hours', 'sleep_hours', 'attendance']]
y = 2*study_hours + 1.5*sleep_hours + 0.5*attendance + np.random.normal(0, 2, n) # Adding some noise
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f'Root Mean Squared Error: {rmse:.4f}')

