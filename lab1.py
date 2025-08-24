import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('Cereals.csv')

x = data['SUGARS'].values.reshape(-1,1)
y = data['RATING'].values

print(data.head())
#print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

#Correlation Heatmap
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Correlation Heatmap of Cereal Features')
plt.show()

#Pairplot for selected features vs Rating
sns.pairplot(data[['CALORIES', 'PROTEIN', 'FAT', 'SODIUM', 'FIBER', 'CARBO', 'SUGARS', 'VITAMINS', 'RATING']], diag_kind='kde')
plt.title('Pairplot of Cereal Nutritional Features & Rating')
plt.show()

#Scatterplot: Calories vs Rating
plt.figure(figsize=(8,6))
sns.scatterplot(x='SUGARS', y='RATING', data=data)
plt.title('Scatter Plot: Cereal Rating vs Sugar')
plt.xlabel('Sugar')
plt.ylabel('Rating')
plt.show()

#Histogram of Ratings
plt.figure(figsize=(8,6))
sns.histplot(data['RATING'], kde=True, bins=20)
plt.title('Histogram of Cereal Rating')
plt.xlabel('Rating')
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
plt.xlabel('Sugar Content')
plt.ylabel('Nutritional Rating')
plt.title('Sugar Content vs Nutritional Rating with Linear Regression')
plt.legend()
plt.grid(True)
plt.show()

#Model coefficients & metrics
print(f'Slope (coefficient): {model.coef_[0]:.2f}')
print(f'Intercept: {model.intercept_:.2f}')
print(f'R2 score: {r2:.2f}')
print(f'MSE: {mse:.2f}')

