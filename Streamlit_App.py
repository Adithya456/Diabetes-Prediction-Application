import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_PORT = os.getenv("DB_PORT", 5432)  # Default to 5432 if not provided

# Database connection function
def get_db_connection():
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            port=os.getenv("DB_PORT")
        )
        return connection
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None
    

# Streamlit Page Configuration
st.set_page_config(page_title="Diabetis Prediction Application", page_icon="Diabetes_App_Logo.png")

# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "About", "Diabetis Prediction", "Batch Prediction", "Feedback Form"]
selected_page = st.sidebar.radio("", pages)

# Home Page
if selected_page == "Home":
    # Streamlit App Layout
    # Create two columns for side-by-side layout
    col1, col2 = st.columns([1, 3])  # Adjust column widths as needed
    # Insert image
    with col1:
        st.image("Diabetes_App_Logo.png", use_container_width=True)
    # Insert title in the second column
    with col2:
        st.markdown("""<h1 style='color: #f5424e; font-size: 50px; font-weight: bold; text-align: center;'>
                    Diabetes Prediction Application</h1>""", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: black;'>By Nagadithya Bathala.</p>", unsafe_allow_html=True)


    # Welcome message
    st.markdown("""
    <div style='text-align: center;'>
        <h3 style='color: #4169E1 ;'>Welcome to the Diabetes Prediction App!</h3>
        <p style='font-size: 18px; color: #555;'>
            Easily predict diabetes risk for individuals or groups using our advanced machine learning models. 
            Your journey towards better health management starts here!
        </p>
    </div>
    """, unsafe_allow_html=True)


# User Inputs
def user_input_features():
    Pregnancies = st.number_input("Number of Pregnancies")
    Glucose = st.number_input("Glucose Level")
    Blood_Pressure = st.number_input("Blood Pressure")
    Skin_Thickness = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin Level")
    BMI = st.number_input("BMI")
    Diabetes_Pedigree_Function = st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("Age")
    
    input_features = np.array([[Pregnancies, Glucose, Blood_Pressure, Skin_Thickness,
                        Insulin, BMI, Diabetes_Pedigree_Function, Age]])
    # Create input data for API
    input_data = {"Pregnancies": Pregnancies, "Glucose": Glucose, "Blood_Pressure": Blood_Pressure,
                  "Skin_Thickness": Skin_Thickness, "Insulin": Insulin, "BMI": BMI,
                  "Diabetes_Pedigree_Function": Diabetes_Pedigree_Function, "Age": Age}
    return input_data

# FastAPI endpoint to fetch the model
SINGLEPREDICTION_API_URL = "https://diabetes-app-backend-825592727774.europe-west2.run.app/predictSingle"
BATCHPREDICTION_API_URL = "https://diabetes-app-backend-825592727774.europe-west2.run.app/predictBatch"

# Home Page
if selected_page == "About":

    # Title for the About Page
    st.markdown("<h1 style='text-align: center; color: #f5424e;'>About Diabetes Prediction App</h1>", unsafe_allow_html=True)

    # Introductory Text
    st.markdown("""
    <div style='text-align: justify; font-size: 18px;'>
    Welcome to the Diabetes Prediction App! This application is designed to assist users in predicting the likelihood of diabetes using advanced machine learning techniques. 
    Whether you're an individual looking for personalized predictions or a healthcare professional analyzing data for multiple users, 
    this app provides seamless solutions to cater to your needs.
    </div>
    """, unsafe_allow_html=True)

    # Section: Key Features
    st.markdown("<h2 style='color: #4169E1;'>Key Features of the App</h2>", unsafe_allow_html=True)

    st.markdown("""
    <ul style='font-size: 18px;'>
        <li><b>Personalized Diabetes Prediction:</b>
            <p>Enter details like age, BMI, glucose level, insulin level, and more to get an instant prediction about diabetes risk.</p>
        </li>
        <li><b>Batch Prediction for Multiple Users:</b>
            <p>Upload a CSV file containing multiple user records, process the data, and download the results as a CSV file.</p>
        </li>
        <li><b>Feedback for Continuous Improvement:</b>
            <p>Share your name, email, and feedback to help us improve the app and provide better service.</p>
        </li>
    </ul>
    """, unsafe_allow_html=True)

    # Section: How It Works
    st.markdown("<h2 style='color: #4169E1 ;'>How It Works</h2>", unsafe_allow_html=True)

    st.markdown("""
    <ol style='font-size: 18px;'>
        <li><b>For Individual Predictions:</b>
            <p>Go to the Diabetes Prediction page, enter your details, and click "Predict" to see the results instantly.</p>
        </li>
        <li><b>For Batch Predictions:</b>
            <p>Navigate to the Batch Prediction page, upload a properly formatted CSV file, and download the predictions.</p>
        </li>
        <li><b>To Provide Feedback:</b>
            <p>Visit the Feedback Form page, fill out the form, and submit your thoughts to help us improve the app.</p>
        </li>
    </ol>
    """, unsafe_allow_html=True)

    # Footer Section
    st.markdown("<h2 style='text-align: center; color: #f5424e;'>Get Started Today!</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; font-size: 18px;'>
    Whether you're tracking your health, assisting others, or analyzing trends, the Diabetes Prediction App is here to make your life easier. 
    Explore our app and take control of diabetes management with confidence.
    </div>
    """, unsafe_allow_html=True)


# Diabetis Prediction
elif selected_page == "Diabetis Prediction":

    # Streamlit App Title
    #st.title('Diabetes Prediction App')
    st.markdown("""<h1 style='color: #f5424e; font-size: 50px; font-weight: bold; text-align: center;'>
                    Diabetes Prediction App</h1>""", unsafe_allow_html=True)

    # Collect input data
    input_data = user_input_features()

    # Prediction Button
    if st.button("Predict"):
        # Send POST request to FastAPI
        response = requests.post(SINGLEPREDICTION_API_URL, json=input_data)
        # Display the prediction
        if response.status_code == 200:
            result = response.json()
            st.success(f"The patient is likely {result['prediction']} with a probalility of {round(result['pred_prob'],2)}")
    
        else:
            st.error("Error making prediction.")


    # Feature Importance
    # Feature Importance using shap_values
    if st.checkbox("Show Feature Importance"):
        # Send POST request to FastAPI
        response = requests.post(SINGLEPREDICTION_API_URL, json=input_data)
        # Fetching Result
        result = response.json()
        # Input Features
        input_features = ['Pregnancies', 'Glucose', 'Blood_Pressure', 'Skin_Thickness',
                            'Insulin', 'BMI', 'Diabetes_Pedigree_Function', 'Age']
        st.subheader("Feature Importance")
        # Create the Bar Plot
        fig, ax = plt.subplots()
        ax.barh(input_features, result['values_'], color='skyblue')
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Features")

        # Display in Streamlit
        st.pyplot(fig)

# Batch Prediction
elif selected_page == "Batch Prediction":
    st.title("Batch Prediction")

    # Upalod file
    uploaded_file = st.file_uploader("Upload a CSV File for Batch Prediction")

    if uploaded_file is not None:
        # API Call for batch prediction
        response = requests.post(BATCHPREDICTION_API_URL, files={"file": uploaded_file.getvalue()})
        if response.status_code == 200:
            result = response.json()
            st.success("Batch prediction completed!")
            batch_data=result["batch_pred"]
            st.download_button(label="Download Predictions", data=batch_data, file_name="predictions.csv",
                               mime="text/csv")
        else:
            st.error("Error making batch prediction")




# Feedback Form
elif selected_page == "Feedback Form":
    st.title("Feedback Form")
    # Feedback Form
    st.header("I Value Your Feedback")
    st.write("Please share your thoughts about my app. Your feedback helps us improve!")

    # Create the form
    user_name = st.text_input("Your Name", placeholder="Enter your name")
    user_email = st.text_input("Your Email", placeholder="Enter your email")
    feedback = st.text_area("Your Feedback", placeholder="Enter your feedback here")

    # Submit button
    if st.button("Submit Feedback"):
        if user_name and feedback:
            connection = get_db_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    # SQL to insert feedback
                    insert_query = """
                    INSERT INTO feedback_table (name, email, feedback)
                    VALUES (%s, %s, %s)
                    """
                    cursor.execute(insert_query, (user_name, user_email, feedback))
                    connection.commit()
                    cursor.close()
                    st.success("Thank you for your feedback!")
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")
                finally:
                    connection.close()
        else:
            st.warning("Please fill out all required fields.")







