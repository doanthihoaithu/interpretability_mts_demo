import streamlit as st
from PIL import Image


image_path = "figures/AUC_PR.png"
image = Image.open(image_path)
st.image(image, caption='AUC_PR of detectors on synthetic datasets')

image_path = "figures/VUS_PR.png"
image = Image.open(image_path)
st.image(image, caption='VUS_PR of detectors on synthetic datasets')

image_path = "figures/INTERPRETABILITY_HIT_2_SCORE.png"
image = Image.open(image_path)
st.image(image, caption='INTERPRETABILITY_HIT_2_SCORE of detectors on synthetic datasets')


image_path = "figures/INTERPRETABILITY_LOG_SCORE.png"
image = Image.open(image_path)
st.image(image, caption='INTERPRETABILITY_LOG_SCORE of detectors on synthetic datasets')