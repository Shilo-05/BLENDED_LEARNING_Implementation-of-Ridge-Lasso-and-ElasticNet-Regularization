# Implementation of Ridge, Lasso, and ElasticNet Regularization for Predicting Car Price

### Developed by: Oswald Shilo
### RegisterNumber: 212223040139

## AIM:
To implement Ridge, Lasso, and ElasticNet regularization models using polynomial features and pipelines to predict car price.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**:  
   - Use pandas for data handling and scikit-learn for preprocessing, modeling, and evaluation tasks.  

2. **Dataset Loading**:  
   - Load the dataset from `encoded_car_data.csv`.  

3. **Feature and Target Selection**:  
   - Separate the dataset into features (X) and target variable (y).  

4. **Data Splitting**:  
   - Split the dataset into training and testing sets with an 80-20 ratio.  

5. **Model and Pipeline Definition**:  
   - Construct pipelines for Ridge, Lasso, and ElasticNet regression models, incorporating polynomial feature transformation with a degree of 2.  

6. **Model Training**:  
   - Train each regression model pipeline using the training data.  

7. **Prediction Generation**:  
   - Use the trained models to predict car prices on the test set.  

8. **Model Evaluation**:  
   - Evaluate each model’s performance by calculating metrics such as Mean Squared Error (MSE) and R² score.  

## Program:
```
# Program to implement Ridge, Lasso, and ElasticNet regularization using pipelines.
# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/encoded_car_data.csv'
df = pd.read_csv(file_path)

# Select relevant features and target variable
X = df.drop(columns=['price'])  # All columns except 'price'
y = df['price']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the models and pipelines
models = {
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5)  # l1_ratio controls L1 vs L2 mix
}

# Iterate over models and evaluate
for name, model in models.items():
    pipeline = Pipeline([
        ("polynomial_features", PolynomialFeatures(degree=2)),
        ("regressor", model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    print(f"\n{name} Regression Results:")
    print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))

```

## Output:
<img width="646" alt="Screenshot 2024-11-17 at 7 57 52 PM" src="https://github.com/user-attachments/assets/28879792-1426-453d-a9d2-1fbeab305a4d">



## Result:
Thus, Ridge, Lasso, and ElasticNet regularization models were implemented successfully to predict the car price and the model's performance was evaluated using R² score and Mean Squared Error.
