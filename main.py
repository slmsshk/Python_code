# https://slmsshk-python-code-main-wbh1t5.streamlitapp.com/

import numpy as np
import time
import streamlit as st
st.set_page_config(page_title="Forecasting Models", page_icon="📈")

for i in range(5):
    st.balloons()
    time.sleep(.5)

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 
def header(url):
     st.markdown(f'<p style="background-color:#f2f3f4;color:#4b5320;font-size:24px;border-radius:2%;text-align:center">{url}</p>', unsafe_allow_html=True)
header("Forecasting")

def header(url):
     st.write(f'<p style="color:#4b5320;font-size:24px;text-align:center">{url}</p>', unsafe_allow_html=True)
header('Navigate to relevant Page')


# name=st.text_input("tell Me your name")
# age=st.number_input("give me your age")
# dob=st.date_input('Enter date')

