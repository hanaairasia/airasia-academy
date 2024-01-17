import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import pickle 

st.write("# Advertising App")
st.write("This app predicts the **Advertising Sale!**")

st.sidebar.header('User Input Platform')

def user_input_features():
    TV = st.sidebar.slider('TV Advertising', 0, 500, 40)
    Radio = st.sidebar.slider('Radio Advertising', 0, 500, 30)
    Newspaper = st.sidebar.slider('Newspaper Advertising', 0, 500, 20)
    data = {'TV': TV,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Advertising.h5", "rb")) #rb: read binary
new_pred = loaded_model.predict(df) # testing (examination)


st.subheader('Prediction')
st.write(new_pred)
