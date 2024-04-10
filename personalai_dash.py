import streamlit as st
import pandas as pd
from personal_ai import *

st.set_page_config(
    layout="wide"
)

personalAI = PersonalAI("IMG_2149.mp4")
personalAI.run(True)

placeholder = st.empty()

count = 0

while True:
    frame,landmarks, ts = personalAI.image_q.get()
    
    if len(landmarks.pose_landmarks)>0:
        frame, elbow_angle = personalAI.find_angle(frame, landmarks, 12, 14, 16, True)
        frame, hip_angle = personalAI.find_angle(frame, landmarks, 11, 23, 25, True)

        #Lógica para as flexões
        if elbow_angle > 150 and hip_angle > 170:
            status = "ready"
            dir = "down"
        if status == "ready":
            if dir == "down" and elbow_angle < 60:
                dir = "up"
                count += 0.5
            if dir == "up" and elbow_angle > 100:
                dir = "down"
                count += 0.5
                
        with placeholder.container():
            st.image(frame)