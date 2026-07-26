import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from ultralytics import YOLO

# --- Page Configuration ---
st.set_page_config(page_title="Warehouse Safety AI", page_icon="🏗️", layout="wide")
st.title("🚨 AI-Powered Warehouse Safety Monitor")
st.markdown("Upload a warehouse CCTV video to detect workers and monitor restricted safety zones in real-time.")

# --- UI Layout ---
col1, col2 = st.columns([1, 3])

with col1:
    st.header("Settings")
    # Added a dropdown to let the user select the AI engine!
    model_choice = st.selectbox("Select AI Engine", ["Base Model (Robust)", "Custom 10-Epoch Model"])
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    uploaded_file = st.file_uploader("Upload Video (MP4)", type=['mp4'])

with col2:
    st.header("Live Feed")
    feed_placeholder = st.empty()
    alert_placeholder = st.empty()

# --- Load Model Based on Choice ---
@st.cache_resource
def load_model(choice):
    if choice == "Base Model (Robust)":
        return YOLO("yolo11n.pt")
    else:
        model_path = "notebooks/runs/trained_models/yolo_worker_safety/weights/best.pt"
        if os.path.exists(model_path):
            return YOLO(model_path)
        st.warning("Custom weights not found. Falling back to YOLO11n.")
        return YOLO("yolo11n.pt")

model = load_model(model_choice)

# --- Video Processing Logic ---
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    zone_polygon = np.array([
        [int(width * 0.1), int(height * 0.95)],  
        [int(width * 0.9), int(height * 0.95)],  
        [int(width * 0.9), int(height * 0.3)],   
        [int(width * 0.1), int(height * 0.3)]    
    ], np.int32)
    zone_polygon = zone_polygon.reshape((-1, 1, 2))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.track(frame, conf=confidence_threshold, persist=True, verbose=False)
        
        zone_color = (0, 255, 0) # Green
        alert_triggered = False
        
        # FIX: Draw boxes if ANY detection exists, regardless of tracker ID
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                feet_x = int((x1 + x2) / 2)
                feet_y = y2
                
                is_inside = cv2.pointPolygonTest(zone_polygon, (float(feet_x), float(feet_y)), False)
                
                if is_inside >= 0:
                    alert_triggered = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) # Red
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2) # Orange
        
        # Handle Alerts
        if alert_triggered:
            zone_color = (0, 0, 255) 
            alert_placeholder.error("ALERT: Worker detected in restricted forklift zone!")
        else:
            alert_placeholder.success("Area Clear.")
            
        cv2.polylines(frame, [zone_polygon], isClosed=True, color=zone_color, thickness=3)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone_polygon], zone_color)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feed_placeholder.image(frame, channels="RGB", use_container_width=True)
        
    cap.release()
    st.info("Video Processing Complete!")