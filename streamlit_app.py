import streamlit as st
from PIL import Image
import tempfile
import os
import json
import cv2

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

    json_path = output_path + ".json"

    # Draw bounding boxes using cv2
    image = cv2.imread(tmp_path)
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    for item in annotations:
        coords = item["pixelCoord"]
        x1 = int(min(pt[0] for pt in coords))
        y1 = int(min(pt[1] for pt in coords))
        x2 = int(max(pt[0] for pt in coords))
        y2 = int(max(pt[1] for pt in coords))
        label = item["entityName"]
        color = (0, 255, 0) if label != "0" else (255, 0, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if label != "0":
            if label.startswith("full_disease_name_"):
                label = label.replace("full_disease_name_", "Disease:")
            elif label.startswith("drug_"):
                label = label.replace("drug_", "Drug:")
            cv2.putText(image, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Annotated Image")
    st.json(data)

    os.remove(tmp_path)
    os.remove(output_path)
    os.remove(json_path)
