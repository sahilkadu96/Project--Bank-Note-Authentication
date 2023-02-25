# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:28:03 2023

@author: Sahil
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title('Bank Note Authentication')

pickle_in = pickle.load(open(r'C:\Users\Sahil\.spyder-py3\classifier.pkl', 'rb'))

uploaded_file = st.file_uploader('upload your input csv file', type = ['csv'])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    pred = pickle_in.predict(input_df)
    
else:
    var = st.slider('Variance', -7.0, 7.0, 0.0, 0.5)
    skew = st.slider('Skewness', -14.0, 13.0, 0.0, 0.5)
    cur = st.slider('Curtosis', -5.0, 18.0, 0.0, 0.5)
    ent = st.slider('Entropy', -9.0, 3.0, 0.0, 0.5)
    pred = pickle_in.predict([[var, skew, cur, ent]])
    
st.write(f"The predicted class is {pred}")