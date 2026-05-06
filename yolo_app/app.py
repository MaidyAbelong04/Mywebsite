import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

st.set_page_config(page_title="AI Object Detector", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title(" Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    results = model.track(
        img, 
        persist=True, 
        conf=conf_threshold, 
        verbose=False
    )

    annotated_frame = results[0].plot()

    if results[0].boxes is not None:
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Objects Count: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]}
    ]
}

webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    async_processing=True,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    video_html_attrs={
        "style": {
            "width": "50%",   
            "max-width": "800px", 
            "margin": "0 auto",   
            "display": "block",
        },
        "controls": False,
        "autoPlay": True,
    },
)
