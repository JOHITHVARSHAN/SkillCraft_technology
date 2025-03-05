import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load the dataset
data = pd.read_csv(r'C:\Users\Sasik\johith\MS_VS_CODE\SkillCraft\house_price_prediction.csv')
print(data.head())

#Train Test Split
X = data.drop('price', axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model Training using Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

#Plotting the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

#Predicting the price of a house
def predict_price(avg_num_rooms, avg_num_bedrooms):
    price = model.predict(np.array([[avg_num_rooms, avg_num_bedrooms]]).reshape(1, -1))
    print(f'Predicted Price for a house with {avg_num_rooms} rooms and {avg_num_bedrooms}: {price[0]}')
    return price[0]