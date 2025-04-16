import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# Custom CSS styling
st.markdown("""
    <style>
    :root {
        --primary: #2c5f2d;
        --secondary: #97bc62;
        --accent: #f5f5f5;
    }
    .main {
        background: linear-gradient(135deg, var(--accent) 0%, #e0f0e9 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: var(--primary);
        text-align: center;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 2rem 0;
    }
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 2rem 0;
        border-left: 5px solid var(--primary);
    }
    .result-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .severity-card {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .cure-card {
        background: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background: var(--primary);
        color: white;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .metric-box {
        background: var(--secondary);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--primary);
        margin-top: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# App Header
st.markdown("<h1>🍃 AI-Powered Brown Spot Disease Detection</h1>", unsafe_allow_html=True)

# Load YOLOv8 model
@st.cache_resource
def load_model():
    try:
        return YOLO('best.pt')  # Update model path
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model()

# Image upload section
with st.container():
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload Leaf Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        st.subheader("🌿 Uploaded Leaf Image")
        st.image(image, use_column_width=True)

        if model and st.button("🔍 Analyze Disease Spread"):
            with st.spinner("🔬 Analyzing leaf health..."):
                results = model(image_bgr)
                annotated_image = results[0].plot()
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                # Results display
                with st.container():
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.subheader("📊 Detection Visualization")
                        st.image(annotated_image_rgb, use_column_width=True)
                    
                    with col2:
                        st.subheader("📈 Health Metrics")
                        detections = results[0].boxes
                        num_spots = len(detections)
                        severity = "Healthy" if num_spots == 0 else \
                                  "Mild" if num_spots <=5 else \
                                  "Moderate" if num_spots <=15 else "Severe"
                        
                        # Severity indicator
                        st.markdown(f"""
                            <div class='metric-box'>
                                <h3>Severity Level</h3>
                                <h2>{severity}</h2>
                                <p>Spots Detected: {num_spots}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed analysis
                        with st.expander("🔍 Detailed Analysis Report"):
                            avg_confidence = np.mean(detections.conf.cpu().numpy()) * 100 if num_spots > 0 else 0
                            st.metric("Average Detection Confidence", f"{avg_confidence:.1f}%")
                            st.progress(avg_confidence/100 if num_spots >0 else 0)
                            
                            st.markdown("""
                                **Severity Classification:**
                                - 0 spots: Healthy
                                - 1-5 spots: Mild infection
                                - 6-15 spots: Moderate infection
                                - 15+ spots: Severe infection
                                """)

                    # Disease management section
                    st.markdown("<div class='severity-card'>", unsafe_allow_html=True)
                    st.subheader("🩺 Recommended Treatment Plan")
                    
                    if severity == "Healthy":
                        st.success("✅ No treatment needed - plant is healthy!")
                    else:
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown("**🌱 Organic Treatment**")
                            st.markdown("""
                                - Neem oil spray (2% solution)
                                - Baking soda mixture (1 tbsp/gallon)
                                - Remove infected leaves
                                - Improve air circulation
                                """)
                        with col_b:
                            st.markdown("**🧪 Chemical Treatment**")
                            st.markdown("""
                                - Chlorothalonil (Daconil)
                                - Mancozeb-based fungicides
                                - Azoxystrobin applications
                                - Follow manufacturer instructions
                                """)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Prevention measures
                    st.markdown("<div class='cure-card'>", unsafe_allow_html=True)
                    st.subheader("🛡️ Prevention Strategies")
                    cols = st.columns(3)
                    with cols[0]:
                        st.markdown("**💧 Cultural Control**")
                        st.markdown("""
                            - Proper irrigation
                            - Crop rotation
                            - Sanitation practices
                            - Balanced fertilization
                            """)
                    with cols[1]:
                        st.markdown("**🔬 Biological Control**")
                        st.markdown("""
                            - Trichoderma species
                            - Pseudomonas fluorescens
                            - Bacillus subtilis
                            - Compost tea sprays
                            """)
                    with cols[2]:
                        st.markdown("**📊 Monitoring**")
                        st.markdown("""
                            - Regular scouting
                            - Weather monitoring
                            - Disease forecasting
                            - Early detection
                            """)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Processing error: {str(e)}")

# Footer
st.markdown("""
    <div class='footer'>
        <p>🌍 Sustainable Agriculture Initiative | 🧠 AI Plant Pathology System v1.0</p>
        <p>📧 Contact: plantdoc@agritech.com | ☎️ Support: +1 (800) 555-AGRI</p>
    </div>
    """, unsafe_allow_html=True)