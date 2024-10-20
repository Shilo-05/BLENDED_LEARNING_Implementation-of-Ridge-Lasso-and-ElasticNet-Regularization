# BLENDED_LEARNING
# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

##### Import Libraries:

Import necessary libraries such as pandas, numpy, matplotlib, seaborn, and relevant modules from sklearn.

##### Load Dataset:

Load the dataset from a specified URL or local file using pandas.


##### Data Preprocessing:

Drop irrelevant columns (e.g., CarName, car_ID).
Convert categorical variables into dummy/indicator variables using one-hot encoding.


##### Define Features and Target:

Split the data into features (X) and the target variable (y), which is typically the price in this case.

##### Split Dataset:

Divide the dataset into training and testing sets using train_test_split, specifying a test size (e.g., 20%).

##### Define Models:

Initialize regression models (e.g., Ridge, Lasso, ElasticNet) with specified hyperparameters (e.g., alpha values).

##### Train Models:

For each model, create a pipeline that includes polynomial feature transformation followed by the regression model.
Fit the pipeline on the training data.

##### Make Predictions:

Use the trained model to make predictions on the testing data.

##### Evaluate Models:

Calculate performance metrics: Mean Squared Error (MSE) and R² Score for each model.

##### Visualize Results:

Create bar plots to visualize the performance metrics for each model, enhancing the plots with annotations for clarity. 

## Program:
```
/*
Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
Developed by: Oswald Shilo
RegisterNumber:  212223040139 
*/
```

```
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Data preprocessing
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and pipelines
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5)
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Create a pipeline with polynomial features and the model
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2)),
        ('regressor', model)
    ])
    
    # Fit the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    predictions = pipeline.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Store results
    results[name] = {'MSE': mse, 'R² Score': r2}

# Print results
for model_name, metrics in results.items():
    print(f"{model_name} - Mean Squared Error: {metrics['MSE']:.2f}, R² Score: {metrics['R² Score']:.2f}")

# Visualization of the results
# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results).T
results_df.reset_index(inplace=True)
results_df.rename(columns={'index': 'Model'}, inplace=True)

# Set the figure size
plt.figure(figsize=(14, 6))

# Bar plot for MSE with a different color palette and complexity
plt.subplot(1, 2, 1)
bar_mse = sns.barplot(x='Model', y='MSE', data=results_df, palette='coolwarm')
plt.title('Mean Squared Error (MSE)', fontsize=16)
plt.ylabel('MSE', fontsize=14)
plt.xticks(rotation=45)

# Adding markers for MSE
for bar in bar_mse.patches:
    bar_value = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, bar_value + 10, f'{bar_value:.2f}', 
             ha='center', va='bottom', fontsize=12)

# Bar plot for R² Score with a different color palette and complexity
plt.subplot(1, 2, 2)
bar_r2 = sns.barplot(x='Model', y='R² Score', data=results_df, palette='plasma')
plt.title('R² Score', fontsize=16)
plt.ylabel('R² Score', fontsize=14)
plt.xticks(rotation=45)

# Adding markers for R² Score
for bar in bar_r2.patches:
    bar_value = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, bar_value + 0.02, f'{bar_value:.2f}', 
             ha='center', va='bottom', fontsize=12)

# Show the plots
plt.tight_layout()
plt.show()

```

## Output:

![image](https://github.com/user-attachments/assets/b044a3d0-4687-420d-abeb-fd71d7aeb5a4)



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
