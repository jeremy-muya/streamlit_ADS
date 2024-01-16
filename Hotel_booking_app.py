#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load your dataset here
csv_path = "C:/Users/HP/Documents/Hotel/hotel_bookings.csv"
df = pd.read_csv(csv_path)

# Load the model
model_path = "C:/Users/HP/Documents/Hotel/best_random_forest_model.pkl"
loaded_model = joblib.load(model_path)

# Disable Matplotlib's global figure warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Streamlit App
st.title("Hotel Booking Prediction with Random Forest Model")

# Display the first few rows of the dataset
st.subheader("Dataset Preview")
st.write(df.head())

# Visualization 1: Bar chart - Comparing stays in week nights, stays in weekend nights and the amount of adults
st.subheader("Stays and Adults Comparison")
fig_stays_adults, ax_stays_adults = plt.subplots(figsize=(10, 6))
sns.barplot(x='stays_in_week_nights', y='stays_in_weekend_nights', hue='adults', data=df, palette="viridis", ax=ax_stays_adults)
ax_stays_adults.set_title("Comparison of Stays in Week Nights, Weekends, and Adults")
ax_stays_adults.set_xlabel("Stays in Week Nights")
ax_stays_adults.set_ylabel("Stays in Weekends")
st.pyplot(fig_stays_adults)
plt.close(fig_stays_adults)

# Visualization 2: Box plot - Arrival date month, adults, is repeat guest
st.subheader("Arrival Date and Guest Comparison")
fig_arrival_guest, ax_arrival_guest = plt.subplots(figsize=(10, 6))
sns.boxplot(x='arrival_date_month', y='adults', hue='is_repeated_guest', data=df, palette="pastel")
ax_arrival_guest.set_title("Comparison of Arrival Date, Adults, and Repeat Guests")
ax_arrival_guest.set_xlabel("Arrival Date Month")
ax_arrival_guest.set_ylabel("Adults")
st.pyplot(fig_arrival_guest)
plt.close(fig_arrival_guest)

# Visualization 3: Bar chart - Arrival date month and adults
st.subheader("Arrival Month and Adults Comparison")
fig_arrival_month, ax_arrival_month = plt.subplots(figsize=(10, 6))
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
sns.barplot(x='arrival_date_month', y='adults', data=df, palette="Set3", order=month_order, ax=ax_arrival_month)
ax_arrival_month.set_title("Comparison of Arrival Month and Adults")
ax_arrival_month.set_xlabel("Arrival Date Month")
ax_arrival_month.set_ylabel("Adults")
st.pyplot(fig_arrival_month)
plt.close(fig_arrival_month)

# Sidebar for user input
st.sidebar.title("User Input for Prediction")

# Function to get user input for a specific feature
def get_user_input(feature):
    return st.sidebar.text_input(f"Enter {feature}", "")

# Iterate through selected features and get user input
selected_features = ['stays_in_week_nights', 'stays_in_weekend_nights', 'adults', 'arrival_date_month']
user_input = {}
for feature in selected_features:
    user_input[feature] = get_user_input(feature)

    # Display the result if the user has provided input for the feature
    if user_input[feature] != "":
        st.subheader(f"Result for {feature} = {user_input[feature]}")
        # Perform the visualization or computation based on the user input
        if feature == 'stays_in_week_nights':
            # Add your specific logic for this feature
            pass
        elif feature == 'stays_in_weekend_nights':
            # Add your specific logic for this feature
            pass
        elif feature == 'adults':
            # Add your specific logic for this feature
            pass
        elif feature == 'arrival_date_month':
            # Add your specific logic for this feature
            pass
        else:
            st.warning("Feature not supported")

        # Add a "Done" button after each feature
        if st.button("Done"):
            st.write("User has completed the input for this feature.")


# In[ ]:




