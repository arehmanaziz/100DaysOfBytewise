import streamlit as st
import pickle
import pandas as pd
import json

# Load the saved model and vectorizer

with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def predict(text):
    # Predicting Text Through Model
    prediction = model.predict([text])[0]

    # Mapping the prediction to sentiment
    sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    return sentiment_mapping[prediction]

st.markdown("<h1 style='text-align: center; color: #45AAFF;'>Sentiment Analysis</h1>", unsafe_allow_html=True)

user_input = st.text_area("Enter The Text")

col1,col2 = st.columns([1,4])


if col1.button("Predict Sentiment"):
    if (user_input!=""):
        sentiment = predict(user_input)
        if (sentiment == "Positive"):
            st.header(f"Predicted Sentiment: {sentiment} 😁")
        
        elif (sentiment == "Neutral") :

            st.header(f"Predicted Sentiment: {sentiment} 😐")

        else:
            st.header(f"Predicted Sentiment: {sentiment} ☹️")




            


        # # Displaying the sentiment analysis results in a pie chart

        # data = {'Sentiment': [sentiment], 'Percentage': [100]}
        # df = pd.DataFrame(data)
        # df['Percentage'] = df['Percentage'].astype(int)

        # st.dataframe(df)
        # st.bar_chart(df)
    
    else:
        st.warning("Please Enter Text Before Pressing The Button")
        
if (col2.button('Reset')):
    user_input=""
    
    

    



