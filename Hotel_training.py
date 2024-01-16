#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib
import os


# In[9]:


directory = "C:/users/HP/Documents/Hotel"
if not os.path.exists(directory):
    os.makedirs(directory)


# In[10]:


# Load the dataset
csv_path = "C:/Users/HP/Documents/Hotel/hotel_bookings.csv"
df = pd.read_csv(csv_path)


# In[11]:


# Explore the Dataset
print(df.head())
print(df.info())
print(df.describe())


# In[12]:


# Missing Values
print(df.isnull().sum())


# In[13]:


# Handle Missing Values
# Fill missing values in 'children' with 0
df['children'].fillna(0, inplace=True)

# Fill missing values in 'country' with a default value
df['country'].fillna('Unknown', inplace=True)

# Drop 'agent' and 'company' columns due to a large number of missing values
df.drop(['agent', 'company'], axis=1, inplace=True)

# Verify that missing values have been handled
print(df.isnull().sum())


# In[14]:


# Preprocess Categorical Variables
columns_to_encode = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type']
df_encoded = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)

# Verify the changes
print(df_encoded.head())


# In[15]:


# Handle Missing Values in Encoded DataFrame
# Assuming 'children' is a numeric feature, fill missing values with the mean
df_encoded['children'].fillna(df_encoded['children'].mean(), inplace=True)

# Handle missing values in 'country_Unknown' (if it exists)
if 'country_Unknown' in df_encoded.columns:
    df_encoded['country_Unknown'].fillna(0, inplace=True)

# Drop any remaining rows with missing values
df_encoded.dropna(inplace=True)

# Display the data types to confirm all are numeric
print(df_encoded.dtypes)


# In[16]:


# Split the Data into Features (X) and Target Variable (y)
X = df_encoded.drop('is_canceled', axis=1)
y = df_encoded['is_canceled']

# Verify the shapes of X and y
print("X shape:", X.shape)
print("y shape:", y.shape)


# In[23]:


# Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the shapes of the resulting sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# In[24]:


# Print data types of the features in the training set
print(X_train.dtypes)


# In[25]:


# Print data type of the target variable
print(y_train.dtypes)

print(y_train.unique())


# In[26]:


# Select categorical columns after splitting
categorical_columns = X_train.select_dtypes(include=['object']).columns


# In[27]:


# One-hot encode categorical variables in both training and testing sets
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)


# In[28]:


# Align the datasets to make sure they have the same columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)


# In[29]:


# Initialize the Random Forest model
random_forest_model = RandomForestClassifier(random_state=42)


# In[30]:


# Fit the model to the training data
random_forest_model.fit(X_train_encoded, y_train)


# In[31]:


# Make Predictions
predictions = random_forest_model.predict(X_test_encoded)


# In[32]:


# Evaluate Model Performance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Other metrics
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


# In[ ]:


# Tune Hyperparameters
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
}

grid_search = GridSearchCV(random_forest_model, param_grid, cv=5)
grid_search.fit(X_train_encoded, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Make predictions with the best model
best_model = grid_search.best_estimator_
best_predictions = best_model.predict(X_test_encoded)


# In[ ]:


# Save the model
model_path = "C:/Users/HP/Documents/Hotel/best_random_forest_model.pkl"
joblib.dump(best_model, model_path)


# In[ ]:




