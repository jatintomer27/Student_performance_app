import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def load_model():
    """
    Load the model from the pickel file
    """
    current_directory = os.getcwd()
    target_path = os.path.abspath(os.path.join(current_directory, "student_lr_final_model.pkl"))
    with open(target_path,'rb') as file:
        model, encorder, scaler = pickle.load(file)
    return model, encorder, scaler

def preprocessing_input_data(data,encoder,scaler):
    data['Extracurricular Activities'] = encoder.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    df_transform = scaler.transform(df.values)
    return df_transform

def predict_data(data):
    model, encorder, scaler = load_model()
    processed_data = preprocessing_input_data(data,encorder,scaler)
    result = model.predict(processed_data)
    return result

def main():

    # To create the app
    st.title("Student performance predection")
    st.write("Enter your data to get for your performance")

    hrs_study = st.number_input("Hours Studied",min_value=1,max_value=10,value=5)
    previous_score = st.number_input("Previous Score",min_value=40,max_value=100,value=70)
    extra_activity = st.selectbox("Extracurricular Activities",["Yes","No"])
    sleep_hrs = st.number_input("Sleep Hours",min_value=4,max_value=10,value=7)
    solved_papers = st.number_input("Number of question paper solved",min_value=0,max_value=10,value=5)

    if st.button("Predict your score"):

        # Do mapping of these variable names with actual names inside the data
        user_data = {
            "Hours Studied":hrs_study,
            "Previous Scores":previous_score,
            "Extracurricular Activities":extra_activity,
            "Sleep Hours":sleep_hrs,
            "Sample Question Papers Practiced":solved_papers,
        }
        predection = predict_data(user_data)
        st.success(f"Your Performance wil be {predection[0]}")
        


# To run this python file as streamlit app:
    
# streamlit run student_performance.py 


if __name__ == "__main__":
    main()
