import streamlit as st
import helper
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

cv = pickle.load(open('cv.pkl','rb'))

model = pickle.load(open('model.pkl','rb'))

st.header('Sales Conversion')

feed = st.text_input('Enter the call feed: ')

if st.button('Find'):
    query = helper.preprocess(feed)
    val_pr = cv.transform([query])
    result = model.predict(val_pr)

    if result==1:
        st.header('Not Converted')
    else:
        st.header('Converted')


