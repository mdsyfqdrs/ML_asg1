#Import relevant packages
import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load the CSV file
data = pd.read_csv('Cereals.csv')
print(data.head())

#Prepare data: x-Fiber vs y-rating
x = data['FIBER'].values.reshape(-1,1)
y = data['RATING'].values

#Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap of the Cereal Features')
plt.show()

#Pairplot for Fiber vs Rating
sns.pairplot(data[['FIBER', 'RATING']], diag_kind='kde')
plt.title('Pairplot of Cereal Rating & Fiber', fontsize = 7, x=0, y=2.025)
plt.show()
plt.show()

#Scatterplot: Fiber vs Rating
plt.figure(figsize=(8,6))
sns.scatterplot(x='FIBER', y='RATING', data=data)
plt.title('Scatter Plot: Cereal Rating vs Fiber')
plt.xlabel('Fiber')
plt.ylabel('Rating')
plt.show()

#Histogram of Ratings
plt.figure(figsize=(8,6))
sns.histplot(data['RATING'], kde=True, bins=20)
plt.title('Histogram of Cereal Rating')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

#Histogram of Fiber
plt.figure(figsize=(8,6))
sns.histplot(data['FIBER'], kde=True, bins=20)
plt.title('Histogram of Cereal Fiber Content')
plt.xlabel('Fiber')
plt.ylabel('Frequency')
plt.show()

#Linear Regression - Create & Train
model = LinearRegression()
model.fit(x,y)

#Predict values
y_pred = model.predict(x)

#Calculate metrics
r2 = r2_score(y, y_pred)
mse = mean_squared_error(y, y_pred)

#Visualization
plt.figure(figsize=(10,6))
plt.scatter(x,y, color='blue', alpha=0.5, label='Data points')
plt.plot(x,y_pred, color='red', label=f'Linear fit (R² = {r2:.2f})')
plt.xlabel('Fiber Content')
plt.ylabel('Nutritional Rating')
plt.title('Fiber Content vs Nutritional Rating with Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

#Model coefficients & metrics
print(f'Slope (coefficient): {model.coef_[0]:.2f}')
print(f'Intercept: {model.intercept_:.2f}')
print(f'R2 score: {r2:.2f}')
print(f'MSE: {mse:.2f}')
