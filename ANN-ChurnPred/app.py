import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle



model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
    
with open('label_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)
    
with open('scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)
    
    
    
    
    
st.title("Customer Churn Prediction")




geography = st.selectbox("Geography", label_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100,)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600, step=1)
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=500000.0, value=50000.0)
tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=10, value=3)
num_of_products = st.slider("Number of Products", 1,4)
has_cr_card = st.radio("Has Credit Card?", [0, 1])  # 1 for Yes, 0 for No
is_active_member = st.radio("Is Active Member?", [0, 1])  # 1 for Yes, 0 for No




input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


geo_encoded = label_encoder_geo([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))


input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df] , axis=1)


input_data_scaled = scalar.transform(input_data)


p = model.predict(input_data_scaled)
pp = p[0][0]

if pp > 0.5 :
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")