import streamlit as st
import joblib
from pathlib import Path
# ==============================
# LOAD MODEL
# ==============================
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(BASE_DIR / "models" / "mobile_price_model.pkl")

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Mobile Price Prediction",
    page_icon="📱",
    layout="wide"
)

# ==============================
# HEADER
# ==============================
st.title("📱 Mobile Price Prediction System")
st.markdown(
    "### Machine Learning based Mobile Price Range Classification"
)

st.info(
    "Enter the mobile specifications below and the trained Random Forest "
    "model will predict its price category."
)

st.divider()

# ==============================
# MOBILE SPECIFICATIONS
# ==============================

st.subheader("🔋 Basic & Performance Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    battery_power = st.number_input(
        "Battery Power (mAh)", 500, 5000, 2000
    )

    ram = st.number_input(
        "RAM (MB)", 256, 8000, 2048
    )

    int_memory = st.number_input(
        "Internal Memory (GB)", 2, 128, 32
    )

with col2:
    clock_speed = st.number_input(
        "Clock Speed (GHz)", 0.1, 4.0, 2.0
    )

    n_cores = st.number_input(
        "Number of Cores", 1, 8, 4
    )

    mobile_wt = st.number_input(
        "Mobile Weight (g)", 80, 250, 150
    )

with col3:
    talk_time = st.number_input(
        "Talk Time (hours)", 2, 25, 12
    )

    m_dep = st.number_input(
        "Mobile Depth (cm)", 0.1, 1.0, 0.5
    )

st.divider()

# ==============================
# CAMERA & DISPLAY
# ==============================

st.subheader("📷 Camera & Display")

col1, col2, col3 = st.columns(3)

with col1:
    fc = st.number_input(
        "Front Camera (MP)", 0, 20, 5
    )

    pc = st.number_input(
        "Primary Camera (MP)", 0, 25, 12
    )

with col2:
    px_height = st.number_input(
        "Pixel Height", 0, 2000, 800
    )

    px_width = st.number_input(
        "Pixel Width", 0, 2000, 1200
    )

with col3:
    sc_h = st.number_input(
        "Screen Height (cm)", 5, 20, 12
    )

    sc_w = st.number_input(
        "Screen Width (cm)", 0, 20, 7
    )

st.divider()

# ==============================
# CONNECTIVITY
# ==============================

st.subheader("📡 Connectivity & Features")

col1, col2, col3, col4 = st.columns(4)

with col1:
    blue = st.selectbox(
        "Bluetooth",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col2:
    dual_sim = st.selectbox(
        "Dual SIM",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col3:
    four_g = st.selectbox(
        "4G",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col4:
    three_g = st.selectbox(
        "3G",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

col1, col2 = st.columns(2)

with col1:
    touch_screen = st.selectbox(
        "Touch Screen",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col2:
    wifi = st.selectbox(
        "WiFi",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

st.divider()

# ==============================
# PREDICTION
# ==============================

st.subheader("🎯 Prediction")

if st.button(
    "🔮 Predict Mobile Price Range",
    use_container_width=True
):

    user_mobile = [[
        battery_power,
        blue,
        clock_speed,
        dual_sim,
        fc,
        four_g,
        int_memory,
        m_dep,
        mobile_wt,
        n_cores,
        pc,
        px_height,
        px_width,
        ram,
        sc_h,
        sc_w,
        talk_time,
        three_g,
        touch_screen,
        wifi
    ]]

    prediction = model.predict(user_mobile)[0]

    categories = {
        0: "Low",
        1: "Medium",
        2: "High",
        3: "Very High"
    }

    category = categories[prediction]

    st.success(
        f"## 📱 Predicted Price Range: {prediction}"
    )

    st.markdown(
        f"### 💰 Price Category: **{category}**"
    )

    st.caption(
        "The prediction is based on the trained Random Forest classification model."
    )

st.divider()

# ==============================
# PROJECT INFORMATION
# ==============================

st.subheader("ℹ️ About This Project")

st.write(
    "This project uses Machine Learning to classify mobile phones into "
    "four different price-range categories based on hardware and feature specifications."
)

st.write(
    "**Algorithm Used:** Random Forest Classifier"
)

st.write(
    "**Target Classes:** 0 = Low, 1 = Medium, 2 = High, 3 = Very High"
)
st.subheader("📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model Accuracy", "88.00%")

with col2:
    st.metric("Test Samples", "400")

with col3:
    st.metric("Algorithm", "Random Forest")
