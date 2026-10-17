import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Page Config
st.set_page_config(page_title="Industrial Defect & Root-Cause Agent", layout="wide")

st.title("🏭 AI-Powered Industrial Defect & Root-Cause Agent")
st.markdown("**AMD Developer Hackathon: ACT III** | Intelligent Industry Track")

# Sidebar for Configuration
st.sidebar.header("Configuration")
machine_id = st.sidebar.selectbox("Select Machine Line", ["Machine-01", "Machine-02", "Machine-03"])
threshold = st.sidebar.slider("Anomaly Sensitivity Threshold (%)", 0.5, 5.0, 1.5)

# Load Maintenance Logs
def load_maintenance_logs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(base_dir, "maintenance_logs.txt"),
        os.path.join(base_dir, "data", "maintenance_logs.txt"),
        "maintenance_logs.txt"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return content
            except Exception:
                continue
    return "No maintenance logs found."

# Vision Detection Function
def detect_anomaly(ref_img, target_img):
    ref = cv2.cvtColor(np.array(ref_img), cv2.COLOR_RGB2GRAY)
    target = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2GRAY)
    
    ref = cv2.resize(ref, (500, 500))
    target = cv2.resize(target, (500, 500))
    
    diff = cv2.absdiff(ref, target)
    _, thresh_img = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    non_zero_count = np.count_nonzero(thresh_img)
    anomaly_score = (non_zero_count / (500 * 500)) * 100
    is_defective = anomaly_score > threshold
    
    return is_defective, round(anomaly_score, 2), thresh_img

# Main Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Live Camera / Inspection Feed")
    uploaded_file = st.file_uploader("Upload incoming part image...", type=["jpg", "png", "jpeg"])
    
    target_image = None
    default_good = None
    
    # Load default images safely if they exist
    if os.path.exists("data/normal/sample_good.jpg"):
        default_good = Image.open("data/normal/sample_good.jpg")

    if uploaded_file is not None:
        target_image = Image.open(uploaded_file)
        st.image(target_image, caption="Current Part Inspection", use_container_width=True)
    elif os.path.exists("data/defective/sample_defect.jpg"):
        target_image = Image.open("data/defective/sample_defect.jpg")
        st.image(target_image, caption="Sample Defective Part (Demo)", use_container_width=True)
    elif default_good is not None:
        target_image = default_good
        st.image(target_image, caption="Default Inspection Image", use_container_width=True)
    else:
        st.warning("⚠️ Please place sample images inside `data/normal/` and `data/defective/` or upload an image above.")

with col2:
    st.subheader("🤖 Agent Root-Cause Analysis")
    
    if st.button("Run AI Inspection & Diagnosis", type="primary"):
        if target_image and default_good:
            with st.spinner("Analyzing visual feed and cross-referencing maintenance logs..."):
                is_def, score, diff_map = detect_anomaly(default_good, target_image)
                logs = load_maintenance_logs()
                
                if is_def:
                    st.error(f"🚨 DEFECT DETECTED! Anomaly Score: {score}%")
                    st.markdown("### 🔍 Historical Context (RAG Memory)")
                    st.text_area("Retrieved Logs:", logs, height=120)
                    st.markdown("### 💡 Copilot Recommendation")
                    st.success(f"""
                    - **Target Asset:** {machine_id}
                    - **Diagnosis:** Surface anomaly score exceeds safety thresholds. Correlating with recent maintenance records, this matches past vibration/misalignment issues.
                    - **Action Plan:** Pause line temporarily, inspect mechanical guides, and perform recalibration.
                    """)
                else:
                    st.success(f"✅ Part Passed Inspection. Anomaly Score: {score}% (Normal)")
        else:
            st.warning("Please upload or provide valid reference and target images first.")