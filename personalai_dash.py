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
    frame,landmarks, ts = personalAI.image_q.get()
    
    if len(landmarks.pose_landmarks)>0:
        frame, elbow_angle = personalAI.find_angle(frame, landmarks, 12, 14, 16, True)
        with placeholder.container():
            st.image(frame)