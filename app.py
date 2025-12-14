# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime
import pandas as pd
import time
import tempfile
import os

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Initialize Session State
# ---------------------------
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        "Incorrect Mask": 0,
        "With Mask": 0,
        "Without Mask": 0,
        "total_detections": 0
    }
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False

if "model" not in st.session_state:
    st.session_state.model = None

if "change_model" not in st.session_state:
    st.session_state.change_model = False


# ---------------------------
# Configuration
# ---------------------------
class_names = ["Incorrect Mask", "With Mask", "Without Mask"]
color_map = {
    0: (255, 165, 0),   # Orange - Incorrect
    1: (0, 255, 0),     # Green - With Mask
    2: (255, 0, 0)      # Red - Without Mask
}

# ---------------------------
# Model Upload Section (Top of Page)
# ---------------------------
st.markdown('<p class="main-header">😷 Face Mask Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Real-Time Mask Compliance Monitoring</p>', unsafe_allow_html=True)

DEFAULT_MODEL_PATH = "fm.keras"


# 1️⃣ Load DEFAULT model automatically
if not st.session_state.model_loaded and not st.session_state.change_model:
    try:
        with st.spinner("Loading default model..."):
            st.session_state.model = tf.keras.models.load_model(
                DEFAULT_MODEL_PATH,
                compile=False
            )
            st.session_state.model_loaded = True

        st.success("✅ Default model loaded")

    except Exception as e:
        st.error(f"❌ Failed to load default model: {e}")
        st.stop()


# 2️⃣ Change model ONLY when user clicks "Change Model"
if st.session_state.change_model:

    st.info("📤 Upload a new trained Keras model")

    uploaded_model = st.file_uploader(
        "Upload Keras Model (.keras or .h5)",
        type=["keras", "h5"]
    )

    if uploaded_model is not None:
        try:
            with st.spinner("Loading new model..."):
                model_bytes = uploaded_model.read()

                suffix = ".keras" if uploaded_model.name.endswith(".keras") else ".h5"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(model_bytes)
                    model_path = tmp.name

                st.session_state.model = tf.keras.models.load_model(
                    model_path,
                    compile=False
                )

                st.session_state.model_loaded = True
                st.session_state.change_model = False  # reset
                os.unlink(model_path)

                st.success("✅ New model loaded successfully!")
                st.rerun()

        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            st.stop()


# 3️⃣ Use loaded model
model = st.session_state.model


# ---------------------------
# Sidebar Configuration
# ---------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/protective-mask.png", width=100)
    st.title("⚙️ Settings")
    
    # Model status
    # st.success("✅ Model Loaded")
    # if st.button("🔄 Change Model"):
    #     st.session_state.model = None
    #     st.session_state.model_loaded = False
    #     st.rerun()
    st.session_state.change_model = st.sidebar.checkbox("🔁 Change Model")

    
    st.divider()
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for predictions"
    )
    
    st.divider()
    
    # Image size settings
    st.subheader("🖼️ Model Input Size")
    
    size_option = st.selectbox(
        "Select image size",
        options=[
            "224 x 224 (Default)",
            "32 x 32",
            "64 x 64",
            "128 x 128",
            "Custom"
        ]
    )
    
    # Default values
    img_width, img_height, channels = 224, 224, 3
    
    if size_option == "32 x 32":
        img_width, img_height = 32, 32
    elif size_option == "64 x 64":
        img_width, img_height = 64, 64
    elif size_option == "128 x 128":
        img_width, img_height = 128, 128
    elif size_option == "Custom":
        img_width = st.slider("Image Width", 16, 512, 224, step=8)
        img_height = st.slider("Image Height", 16, 512, 224, step=8)
        channels = st.selectbox("Channels", [1, 3])
    
    st.info(f"📐 Input Size: {img_width} x {img_height} x {channels}")
    
    st.divider()
    
    # Detection settings
    st.subheader("🎯 Detection Options")
    show_confidence = st.checkbox("Show Confidence Score", value=True)
    show_timestamp = st.checkbox("Show Timestamp", value=True)
    
    # Camera settings
    st.subheader("📹 Camera Settings")
    camera_resolution = st.selectbox(
        "Resolution",
        ["640x480", "1280x720", "1920x1080"],
        index=0
    )
    
    # Face detection settings
    use_face_detection = st.checkbox("Enable Face Detection", value=True, 
                                      help="Detect faces first, then classify. Disable to classify entire image.")
    
    st.divider()
    
    # Statistics
    st.subheader("📊 Detection Statistics")
    st.metric("Total Detections", st.session_state.stats['total_detections'])
    st.metric("With Mask", st.session_state.stats['With Mask'], 
              f"{st.session_state.stats['With Mask']/max(st.session_state.stats['total_detections'],1)*100:.1f}%")
    st.metric("Without Mask", st.session_state.stats['Without Mask'],
              f"{st.session_state.stats['Without Mask']/max(st.session_state.stats['total_detections'],1)*100:.1f}%")
    st.metric("Incorrect Mask", st.session_state.stats['Incorrect Mask'],
              f"{st.session_state.stats['Incorrect Mask']/max(st.session_state.stats['total_detections'],1)*100:.1f}%")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset"):
            st.session_state.stats = {
                "Incorrect Mask": 0,
                "With Mask": 0,
                "Without Mask": 0,
                "total_detections": 0
            }
            st.session_state.detection_history = []
            st.rerun()
    
    with col2:
        if st.button("📥 Export"):
            if st.session_state.detection_history:
                df = pd.DataFrame(st.session_state.detection_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data to export")

# ---------------------------
# Load Face Detector
# ---------------------------
@st.cache_resource
def load_face_detector():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    return face_cascade

face_cascade = load_face_detector()

# ---------------------------
# Helper Functions
# ---------------------------
def detect_faces(image):
    """Detect faces in the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    return faces

def predict_mask(image, img_width, img_height, channels=3):
    """Predict mask status for the image"""
    try:
        # Resize image
        img = cv2.resize(image, (img_width, img_height))
        
        # Handle grayscale models
        if channels == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, axis=-1)
        
        # Normalize & add batch dimension
        img_array = np.expand_dims(img / 255.0, axis=0)
        
        # Prediction
        preds = model.predict(img_array, verbose=0)
        
        pred_class = np.argmax(preds)
        confidence = np.max(preds)
        
        return pred_class, confidence
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0.0

def update_statistics(pred_class, confidence):
    """Update detection statistics"""
    if confidence >= confidence_threshold and pred_class is not None:
        st.session_state.stats[class_names[pred_class]] += 1
        st.session_state.stats['total_detections'] += 1
        
        # Add to history
        st.session_state.detection_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'detection': class_names[pred_class],
            'confidence': f"{confidence*100:.2f}%"
        })

def annotate_image(image, faces=None, update_stats=False):
    """Annotate image with predictions"""
    annotated = image.copy()
    h, w = image.shape[:2]
    detections = []  # Store detections for batch stats update
    
    if use_face_detection and faces is not None and len(faces) > 0:
        # Process each detected face
        for (x, y, w_face, h_face) in faces:
            # Extract face region
            face_roi = image[y:y+h_face, x:x+w_face]
            
            # Predict
            pred_class, confidence = predict_mask(face_roi, img_width, img_height, channels)
            
            if pred_class is not None and confidence >= confidence_threshold:
                # Store for stats update
                detections.append((pred_class, confidence))
                
                # Draw rectangle around face
                color = color_map[pred_class]
                cv2.rectangle(annotated, (x, y), (x+w_face, y+h_face), color, 3)
                
                # Prepare label
                label = class_names[pred_class]
                if show_confidence:
                    label += f": {confidence*100:.1f}%"
                
                # Draw label background
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x, y-30), (x+label_size[0]+10, y), color, -1)
                
                # Draw label text
                cv2.putText(annotated, label, (x+5, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # Process entire image
        pred_class, confidence = predict_mask(image, img_width, img_height, channels)
        
        if pred_class is not None and confidence >= confidence_threshold:
            # Store for stats update
            detections.append((pred_class, confidence))
            
            color = color_map[pred_class]
            label = class_names[pred_class]
            if show_confidence:
                label += f": {confidence*100:.1f}%"
            
            # Draw on top of image
            cv2.putText(annotated, label, (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Add timestamp if enabled
    if show_timestamp:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(annotated, timestamp, (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Update statistics if requested (only once per frame/image)
    if update_stats:
        for pred_class, confidence in detections:
            update_statistics(pred_class, confidence)
    
    return annotated, len(detections)

# ---------------------------
# Main Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📷 Upload Image", "🎥 Live Camera", "📊 Analytics", "ℹ️ About"])

# ---------------------------
# Tab 1: Upload Image
# ---------------------------
with tab1:
    st.header("Upload Image for Detection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload an image containing faces"
        )
        
        if uploaded_file is not None:
            try:
                image = np.array(Image.open(uploaded_file).convert("RGB"))
                
                # Display original image
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
                
                if st.button("🔍 Detect Mask", key="detect_btn"):
                    with st.spinner("Analyzing image..."):
                        # Detect faces
                        faces = detect_faces(image) if use_face_detection else None
                        
                        # Annotate image and update stats
                        annotated, num_detections = annotate_image(image, faces, update_stats=True)
                        
                        # Store in session state for display
                        st.session_state.last_annotated = annotated
                        st.session_state.last_num_detections = num_detections
                        st.session_state.last_num_faces = len(faces) if faces is not None else 0
                        
                        st.rerun()
            
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    with col2:
        if uploaded_file is not None and 'last_annotated' in st.session_state:
            st.subheader("Detection Result")
            st.image(st.session_state.last_annotated, use_container_width=True)
            
            # Show detection summary
            if use_face_detection:
                st.success(f"✅ Detected {st.session_state.last_num_faces} face(s)")
                st.info(f"📊 {st.session_state.last_num_detections} detection(s) above confidence threshold")
            else:
                if st.session_state.last_num_detections > 0:
                    st.success(f"✅ Image classified successfully")
                else:
                    st.warning("⚠️ Confidence below threshold")

# ---------------------------
# Tab 2: Live Camera
# ---------------------------
with tab2:
    st.header("Live Camera Detection")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        camera_placeholder = st.empty()
        status_placeholder = st.empty()
    
    with col2:
        st.subheader("Live Stats")
        fps_text = st.empty()
        detection_text = st.empty()
        faces_text = st.empty()
    
    run_camera = st.checkbox("🎥 Start Camera", key="camera_toggle")
    
    if run_camera:
        # Parse resolution
        width, height = map(int, camera_resolution.split('x'))
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Failed to access camera. Please check camera permissions.")
            st.stop()
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        prev_time = time.time()
        frame_count = 0
        
        status_placeholder.info("📹 Camera active - Processing frames...")
        
        try:
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    status_placeholder.error("❌ Failed to read frame")
                    break
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
                prev_time = curr_time
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect and annotate (update stats every 10 frames to reduce overhead)
                faces = detect_faces(frame_rgb) if use_face_detection else None
                update_stats_flag = (frame_count % 10 == 0)
                annotated_frame, num_detections = annotate_image(frame_rgb, faces, update_stats=update_stats_flag)
                
                # Add FPS counter
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                           (annotated_frame.shape[1]-120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display frame
                camera_placeholder.image(annotated_frame, channels="RGB", use_container_width=True)
                
                # Update live stats
                fps_text.metric("FPS", f"{fps:.1f}")
                if use_face_detection and faces is not None:
                    faces_text.metric("Faces Detected", len(faces))
                detection_text.metric("Active Detections", num_detections)
                
                frame_count += 1
                time.sleep(0.01)  # Small delay to prevent overwhelming CPU
                
        except Exception as e:
            st.error(f"Camera error: {e}")
        
        finally:
            cap.release()
            status_placeholder.success("✅ Camera stopped")
    else:
        st.info("👆 Check the box above to start camera detection")

# ---------------------------
# Tab 3: Analytics
# ---------------------------
with tab3:
    st.header("Detection Analytics")
    
    if st.session_state.stats['total_detections'] > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        total = st.session_state.stats['total_detections']
        
        with col1:
            st.metric("Total Detections", total)
        with col2:
            count = st.session_state.stats['With Mask']
            pct = (count / total * 100)
            st.metric("With Mask", count, f"{pct:.1f}%")
        with col3:
            count = st.session_state.stats['Without Mask']
            pct = (count / total * 100)
            st.metric("Without Mask", count, f"{pct:.1f}%")
        with col4:
            count = st.session_state.stats['Incorrect Mask']
            pct = (count / total * 100)
            st.metric("Incorrect Mask", count, f"{pct:.1f}%")
        
        # Chart
        st.subheader("Detection Distribution")
        chart_data = pd.DataFrame({
            'Category': ['With Mask', 'Without Mask', 'Incorrect Mask'],
            'Count': [
                st.session_state.stats['With Mask'],
                st.session_state.stats['Without Mask'],
                st.session_state.stats['Incorrect Mask']
            ]
        })
        st.bar_chart(chart_data.set_index('Category'))
        
        # Recent detections
        if st.session_state.detection_history:
            st.subheader("Recent Detections")
            recent_count = st.slider("Show last N detections", 5, 50, 10)
            df = pd.DataFrame(st.session_state.detection_history[-recent_count:][::-1])
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("📊 No detection data available yet. Start detecting to see analytics!")

# ---------------------------
# Tab 4: About
# ---------------------------
with tab4:
    st.header("About This Application")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This application uses deep learning to detect whether people are wearing face masks correctly,
        helping organizations maintain safety protocols and compliance.
        
        ### 🚀 Features
        - **Real-time Detection**: Live camera feed analysis with FPS counter
        - **Batch Processing**: Upload and analyze images
        - **Face Detection**: Automatic face localization using Haar Cascades
        - **Statistics Tracking**: Comprehensive analytics and history
        - **Customizable Settings**: Adjust confidence, resolution, and model input size
        - **Export Data**: Download detection history as CSV
        
        ### 📊 Detection Categories
        1. **With Mask** 🟢: Person wearing mask correctly
        2. **Without Mask** 🔴: Person not wearing a mask
        3. **Incorrect Mask** 🟠: Person wearing mask incorrectly
        
        ### 🔧 Technology Stack
        - **Framework**: Streamlit
        - **ML Model**: TensorFlow/Keras
        - **Computer Vision**: OpenCV
        - **Face Detection**: Haar Cascade Classifier
        
        ### 📝 Usage Instructions
        1. Upload your trained Keras model (.keras or .h5 format)
        2. Configure detection settings in the sidebar
        3. Choose between:
           - **Upload Image**: Analyze static images
           - **Live Camera**: Real-time video detection
        4. View statistics and download detection history
        
        ### ⚙️ Configuration Options
        - **Confidence Threshold**: Minimum prediction confidence (0.0 - 1.0)
        - **Model Input Size**: Match your model's expected input dimensions
        - **Face Detection**: Enable/disable automatic face localization
        - **Camera Resolution**: Adjust video quality (640x480 to 1920x1080)
        
        ### 💡 Tips for Best Results
        - Ensure good lighting conditions
        - Position faces clearly in frame
        - Match model input size to your training configuration
        - Adjust confidence threshold based on your requirements
        - Use face detection for multi-person scenarios
        """)
    
    with col2:
        st.subheader("📈 System Info")
        st.info(f"**Model Status**: {'✅ Loaded' if st.session_state.model_loaded else '❌ Not Loaded'}")
        st.info(f"**TensorFlow**: {tf.__version__}")
        st.info(f"**OpenCV**: {cv2.__version__}")
        
        st.subheader("🎨 Color Legend")
        st.markdown("🟢 **Green** - With Mask")
        st.markdown("🔴 **Red** - Without Mask")
        st.markdown("🟠 **Orange** - Incorrect Mask")
        
        st.subheader("⚠️ Requirements")
        st.markdown("""
        - Trained Keras model file
        - Webcam for live detection
        - Supported formats: JPG, PNG
        """)
    
    st.success("✅ Application running successfully!")

# ---------------------------
# Footer
# ---------------------------
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Face Mask Detection System v2.1 | Powered by TensorFlow & Streamlit</p>
        <p>© 2025 | Built for Safety & Compliance</p>
    </div>
""", unsafe_allow_html=True)
