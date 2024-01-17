import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# Advertising App")
st.write("This app predicts the **Advertising Sale!**")

st.sidebar.header('User Input Platform')

def user_input_features():
    TV = st.sidebar.slider('TV Advertising', 0, 500, 40)
    Radio = st.sidebar.slider('Radio Advertising', 0, 500, 30)
    Newspaper = st.sidebar.slider('Newspaper Advertising', 0, 500, 20)
    data = {'TV Advertising': TV,
            'Radio Advertising': Radio,
            'Newspaper Advertising': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("Advertising.h5", "rb")) #rb: read binary
new_pred = loaded_model.predict(X_test) # testing (examination)
dfnew_pred = pd.DataFrame({'Actual': y_test, 'Predicted': new_pred})
dfnew_pred

prediction = modelGaussianIris.predict(df)
prediction_proba = modelGaussianIris.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
