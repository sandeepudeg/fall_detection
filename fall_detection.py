import streamlit as st
import cv2
import requests
import numpy as np
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# --- Configuration ---
API_KEY = "yvRcl86SKiLbXmEnKypF"
PROJECT_ID = "fall-detection-acou5"
MODEL_VERSION = 1
INFERENCE_URL = f"https://detect.roboflow.com/{PROJECT_ID}/{MODEL_VERSION}?api_key={API_KEY}"

st.set_page_config(page_title="Fall Detection Dashboard", layout="wide")

# --- Logic: Roboflow Inference ---
def infer_frame(frame):
    """Sends frame to Roboflow API."""
    _, img_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(INFERENCE_URL, files={'file': img_encoded.tobytes()})
    return response.json() if response.status_code == 200 else None

def draw_predictions(frame, predictions, conf_threshold):
    """Draws boxes on the frame (expects BGR for CV2, converts to RGB later)."""
    for pred in predictions:
        if pred["confidence"] >= conf_threshold:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            class_name = pred["class"].lower()
            
            # Red for Fall, Green for No Fall
            is_fall = "fall" in class_name and "no" not in class_name
            color = (0, 0, 255) if is_fall else (0, 255, 0) # BGR for CV2
            
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"{class_name} {pred['confidence']:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# --- WebRTC Worker for Cloud Webcam ---
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.conf_threshold = 0.3
        self.frame_skip = 5
        self.count = 0
        self.last_preds = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.count += 1
        
        # Only call API every N frames to avoid lag
        if self.count % self.frame_skip == 0:
            data = infer_frame(img)
            self.last_preds = data.get("predictions", []) if data else []
        
        img = draw_predictions(img, self.last_preds, self.conf_threshold)
        
        # Return as an AV frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Main UI ---
st.title("🏃‍♂️ Fall Detection Dashboard")

st.sidebar.title("Settings")
source_radio = st.sidebar.radio("Select Input Source:", ("Video Upload", "Live Webcam"))
conf_level = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.30)
frame_stride = st.sidebar.number_input("Process every X frames", min_value=1, value=5)

if source_radio == "Video Upload":
    uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        frame_count = 0
        last_preds = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_count % frame_stride == 0:
                data = infer_frame(frame)
                last_preds = data.get("predictions", []) if data else []
            
            frame = draw_predictions(frame, last_preds, conf_level)
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            frame_count += 1
        cap.release()
    else:
        st.info("Please upload a video file to begin.")

elif source_radio == "Live Webcam":
    st.info("Streaming through browser WebRTC. Click 'Start' below.")
    
    # This component captures the user's browser camera
    ctx = webrtc_streamer(
        key="fall-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={ 
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
    )
    
    # Link sidebar settings to the processor
    if ctx.video_processor:
        ctx.video_processor.conf_threshold = conf_level
        ctx.video_processor.frame_skip = frame_stride