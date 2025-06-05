import streamlit as st
from PIL import Image
import tempfile
import os

from main import process_image_with_ocr

st.title("BioBERT NER Tagger")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    fields = []
    mode = 1
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    output_path = tmp_path + "_annotated.png"
    data = process_image_with_ocr(tmp_path, output_path, fields, mode)

    st.image(Image.open(output_path), caption="Annotated Image")
    st.json(data)

    os.remove(tmp_path)
    os.remove(output_path)
