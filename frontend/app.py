import streamlit as st
import requests

st.title("Text Classifier")
text = st.text_area("Enter text")

if st.button("Predict"):
    response = requests.post("http://backend:8000/predict", json={"text": text})
    result = response.json()
    st.write(f"Predicted label: **{result['label']}**")

# try:
#     response = requests.post(
#         "http://backend:8000/predict", json={"text": text}, timeout=5
#     )
#     result = response.json()
#     st.write(f"Predicted label: **{result['label']}**")
# except requests.exceptions.ConnectionError:
#     st.warning("Backend is still starting up — please wait a moment and try again.")
