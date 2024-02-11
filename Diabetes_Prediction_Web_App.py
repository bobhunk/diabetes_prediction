import numpy as np
import pickle
import streamlit as st


with open('C:/Users/robert.tumushiime/AI_AND_ML_ASSIGNMENT/logistic_regression_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

#creating a funtion for prediction
    
def diabetes_prediction(input_data):

    
    input_data = (5,166,72,19,175,25.8,0.587,51)

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]==0):
        return ('The person is not diabetic')
    else:
        return ('The person is diabetic')
    
def main():
    
    st.title('Diabetes Prediction Web App')

    #User Input
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction= st.text_input('DiabetesPedigreeFunction value')
    Age = st.text_input('Age of person')

    #code for prediction
    diagnosis = ''

    #creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

    st.success(diagnosis)



if __name__ == '__main__':
    main()