import streamlit as st
import pandas as pd
from personal_ai import *

st.set_page_config(
    layout="wide"
)

personalAI = PersonalAI("IMG_2149.mp4")
personalAI.run(True)

placeholder = st.empty()
while True:
    frame = personalAI.image_q.get()
    with placeholder.container():
        st.image(frame)