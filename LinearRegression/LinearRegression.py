#import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression


#Load Data Sets
df = pd.read_csv('LinearRegression/Salary_Data.csv')
print(df.head())

#Describe Data
print(df.describe())

"""#Data Distribution
plt.title('Salary Distribution Plot')
sns.distplot(df['Salary'])
plt.show()"""

#Relationship Between Salary and Experience
plt.scatter(df['YearsExperience'], df['Salary'], color = 'lightcoral')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.box(False)
plt.show()

#Splitting variables
X = df.iloc[:, :1]
y = df.iloc[:, 1:]

#Splitting dataset into test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

#Prediction reasult
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

#Prediction on training set
plt.scatter(X_train, y_train)

# Prediction on test set
plt.scatter(X_test, y_test, color = 'lightcoral')
plt.plot(X_train, y_pred_train, color = 'firebrick')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['X_train/Pred(y_test)', 'X_train/y_train'], title = 'Sal/Exp', loc='best', facecolor='white')
plt.box(False)
plt.show()

# Regressor coefficients and intercept
print(f'Coefficient: {model.coef_}')
print(f'Intercept: {model.intercept_}')