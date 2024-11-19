import streamlit as st
import pickle
import pandas as pd
from preprocessor import Preprocessor
import streamlit_lottie 
import nltk




# Load the entire pipeline
with open('pipe_lstm.pkl', 'rb') as f:
    pipe = pickle.load(f)

def run():
    # Add an animation
    st.lottie("https://lottie.host/8285de03-31b0-4fce-98d7-e9e6fd886268/diocPQmJal.json", width=200, height=200)
    st.title("Fake News Detection")
    st.text("Predict whether a news article is Real or Fake")
    
    user_input = st.text_area('Enter news content below:', placeholder='Input news content here...')
    st.text("")
    
    if st.button("Predict"):
        # Make prediction using the pipeline directly
        prediction = pipe.predict(pd.Series([user_input]))[0]  # Get the prediction from the pipeline
        
        # Interpret prediction result
        if prediction >= 0.7:
            output = 'Real 👍'
        else:
            output = 'Fake 👎'
        
        label = f'The news article is likely {output}'
        st.success(label)

    # Footer
    st.markdown("**Created with ❤️ by Ayan**")

if __name__ == "__main__":
    run()

    