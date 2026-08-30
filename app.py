import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Freshness Classifier", page_icon="🍎")

st.title("🍎 Fruit Freshness Classifier")
st.write("Upload an image of a fruit to check if it's fresh or rotten.")

API_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Freshness"):
        with st.spinner("Analyzing..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post(API_URL, files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)})

            if response.status_code == 200:
                result = response.json()
                label = result["prediction"]
                confidence = result["confidence"]

                if label == "fresh":
                    st.success(f"✅ Fresh! (Confidence: {confidence*100:.2f}%)")
                else:
                    st.error(f"❌ Rotten! (Confidence: {confidence*100:.2f}%)")
            else:
                st.error("Error connecting to API. Make sure the backend is running.")