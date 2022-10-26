import numpy as np
import time
import streamlit as st
st.set_page_config(page_title="Forecasting Models", page_icon="📈")

for i in range(5):
    st.balloons()
    time.sleep(.5)



st.write(f'Streamlit version {st.__version__}')
st.write(f'Numpy version {np.__version__}_')

# st.sidebar()