import streamlit as st
import cv2
import requests
import numpy as np
import tempfile

# --- Roboflow Config ---
API_KEY = "yvRcl86SKiLbXmEnKypF"
PROJECT_ID = "fall-detection-acou5"
MODEL_VERSION = 1
INFERENCE_URL = f"https://detect.roboflow.com/{PROJECT_ID}/{MODEL_VERSION}?api_key={API_KEY}"

st.set_page_config(page_title="Fall Detection Dashboard", layout="wide")

# --- Sidebar ---
st.sidebar.title("Settings")

# This is the part that was missing in your screenshot!
source_radio = st.sidebar.radio("Select Input Source:", ("Video Upload", "Live Webcam"))

conf_level = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.30)
frame_skip = st.sidebar.number_input("Process every X frames", min_value=1, value=3)

def infer_frame(frame):
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(INFERENCE_URL, files={'file': img_encoded.tobytes()})
    return response.json() if response.status_code == 200 else None

def run_detection(video_source):
    cap = cv2.VideoCapture(video_source)
    st_frame = st.empty()
    stop_btn = st.sidebar.button("Stop Processing")
    
    frame_count = 0
    last_preds = []

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret: break

        if frame_count % frame_skip == 0:
            data = infer_frame(frame)
            last_preds = data.get("predictions", []) if data else []

        for pred in last_preds:
            if pred["confidence"] >= conf_level:
                x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                is_fall = "fall" in pred["class"].lower() and "no" not in pred["class"].lower()
                color = (255, 0, 0) if is_fall else (0, 255, 0) 
                
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{pred['class']} {pred['confidence']:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
        frame_count += 1
    cap.release()

# --- Main Page ---
st.title("🏃‍♂️ Fall Detection Dashboard")

if source_radio == "Video Upload":
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        run_detection(tfile.name)
    else:
        st.info("Please upload a video file to begin.")

elif source_radio == "Live Webcam":
    if st.sidebar.checkbox("Start Camera"):
        run_detection(0) # 0 is the default webcam
    else:
        st.info("Check 'Start Camera' in the sidebar to begin.")