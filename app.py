# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 20:54:55 2021

@author: tanish
"""

import numpy as np
import tensorflow as tf
import PIL
import streamlit as st

import cv2
from PIL import Image, ImageOps


html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Famous Personalities Image Classifier </h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
st.set_option('deprecation.showfileUploaderEncoding',False)
@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('my_model.h5')
    return model

model=load_model()
st.write("""
         Personalities include - Anushka_Sharma, Barack_Obama,
             Bill_Gates, Dalai_Lama,
              Indira_Nooyi, Melinda_Gates,
              Narendra_Modi, Sundar_Pichai,
              Vikas_Khanna, Virat_Kohli
         """
         )
uploaded_file=st.file_uploader("Please upload an image",type=["jpg","png"])
from PIL import Image,ImageOps


if uploaded_file is None:
    st.text("")
else:
    result=0
     # To read file as bytes:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if img is not None:
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       faces = face_cascade.detectMultiScale(gray, 1.3, 5)

       cropped_faces = []
       for (x,y,w,h) in faces:
               
               roi_color = img[y:y+h, x:x+w]
               cropped_faces.append(roi_color)
       
    
    for img in cropped_faces:
       scalled_raw_img = cv2.resize(img, (32, 32))
       
       scalled_raw_img = np.array(scalled_raw_img).reshape(1,32,32,3).astype(float)
       result=model.predict_classes(scalled_raw_img)

    class_names=["Anushka_Sharma", "Barack_Obama",
              "Bill_Gates", "Dalai_Lama",
              "Indira_Nooyi", "Melinda_Gates",
              "Narendra_Modi", "Sundar_Pichai",
              "Vikas_Khanna", "Virat_Kohli"]
    try:
        st.text( "The image is of "+class_names[result[0]])
    except:
        st.text("Face not detectable")


