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

## REGULARIZATION TECHNIQUES
#Linear Regression = baseline model (best straight line fit).
#Ridge = keeps slope smaller to avoid overfitting.
#Lasso = like Ridge but can set slope to zero if alpha is large (simplifies the model).

from sklearn.linear_model import Ridge, Lasso

# Note: Already have  x(FIBER) & y(RATING) prepared earlier, Line 13-14

# RIDGE REGRESSION
#Ridge = Linear Regression + L2 penalty
#The L2 penalty 'shrinks' the slope towards zero
#Helps avoid overly large coefficients - reduce overfitting
ridge = Ridge(alpha=10.0) #alpha = regularization strength (higher = stronger penalty)
ridge.fit(x,y)
ridge_pred = ridge.predict(x)

# EVALUATE RIDGE
ridge_r2 = r2_score(y, ridge_pred) # R^2 score = goodness of fit
ridge_mse = mean_squared_error(y, ridge_pred) ## Mean Squared Error
print("\n--- Ridge Regression ---")
print(f"Slope: {ridge.coef_[0]:.2f}")         # slope (coefficient for Fiber)
print(f"Intercept: {ridge.intercept_:.2f}")   # intercept (constant term)
print(f"R2 score: {ridge_r2:.2f}")            # how well the model explains the data
print(f"MSE: {ridge_mse:.2f}")                # error between predicted vs actual

# LASSO REGRESSION
#Lasso = Linear Regression + L1 penalty
#The L1 penalty can shrink slope all the way to zero (feature selection)
lasso = Lasso(alpha=10.0) # alpha = regularization strength
lasso.fit(x,y)
lasso_pred = lasso.predict(x)

# EVALUATE LASSO
lasso_r2 = r2_score(y, lasso_pred)
lasso_mse = mean_squared_error(y, lasso_pred)
print("\n--- Lasso Regression ---")
print(f"Slope: {lasso.coef_[0]:.2f}")         # slope for Fiber
print(f"Intercept: {lasso.intercept_:.2f}")   # intercept
print(f"R2 score: {lasso_r2:.2f}")            # model performance
print(f"MSE: {lasso_mse:.2f}")                # error

# VISUALIZATION
plt.figure(figsize=(10,6))
plt.scatter(x,y, color='blue', alpha=0.5, label='Data points')
plt.plot(x,ridge_pred, color="yellow", label=f'Ridge(R2={ridge_r2:.2f})') #plot ridge line
plt.plot(x, lasso_pred, color='green',label=f'Lasso(R2={lasso_r2:.2f})') #plot lasso line
plt.plot(x, y_pred, color='pink',label=f'Linear(R2={r2:.2f})') #plot linear regression line

plt.xlabel('Fiber')
plt.ylabel('Rating')
plt.title('Fiber vs Rating with Linear, Ridge, and Lasso Regression')
plt.legend()
plt.grid(True)
plt.show()



## ASSIGNMENT X 2: MULTIPLE LINEAR REGRESSION
data2 = pd.read_csv('Cereals.csv')

feature_cols = ['FIBER', 'PROTEIN']
x2=data2[feature_cols]
y2=data2[['RATING']]

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(data2[['FIBER','PROTEIN','RATING']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap: Fiber, Protein & Rating')
plt.show()

# Pairplot
sns.pairplot(data2[['FIBER','PROTEIN','RATING']], diag_kind='kde')
plt.suptitle('Pairplot of Cereal Rating vs Fiber & Protein', y=1.02)
plt.show()

# Scatterplots
plt.figure(figsize=(8,6))
sns.scatterplot(x='FIBER', y='RATING', data=data2, label='Fiber')
sns.scatterplot(x='PROTEIN', y='RATING', data=data2, label='Protein')
plt.xlabel('Nutritional Content')
plt.ylabel('Rating')
plt.title('Cereal Rating vs Fiber & Protein')
plt.legend()
plt.show()

#Multiple Linear Regression
model2 = LinearRegression()
model2.fit(x2,y2)

y2_pred = model2.predict(x2) # predict values

# metrics
r2_2 = r2_score(y2, y2_pred)
mse2 = mean_squared_error(y2, y2_pred)

print("Intercept:" , model2.intercept_)
print("Slope:" , model2.coef_)
print("R2 score:" , r2_2)
print("MSE:" , mse2)

# Visualization: Actual vs Predicted
plt.figure(figsize=(8,6))
plt.scatter(y2, y2_pred, color='blue', alpha=0.5)
plt.axline((0,0), slope=1, color='red', linewidth=2)
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Multiple Linear Regression: Actual vs Predicted Ratings')
plt.grid(True)
plt.show()