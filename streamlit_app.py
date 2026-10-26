"""
streamlit_app.py - Diabetes Risk Predictor (frontend)
 
Yeh app model.pkl load NAHI karti. Sirf HTTP ke zariye FastAPI /predict
route se baat karti hai.
 
Run:
    1) python -m uvicorn main:app --reload
    2) streamlit run streamlit_app.py
"""
 
import requests
import streamlit as st
 
# ── 1. Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
API_URL = "https://cavalry-tying-essential.ngrok-free.dev"
 
# ── 2. Styling ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
 
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stApp {
        background: #f5f8f7;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1150px;
    }
 
    /* Hero */
    .hero {
        background: linear-gradient(120deg, #0f3d3e 0%, #14595a 100%);
        color: #ffffff;
        padding: 2.2rem 2.4rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 0.4rem 0;
        padding: 0;
    }
    .hero p {
        color: #cfe5e2;
        font-size: 1.02rem;
        margin: 0;
        max-width: 640px;
        line-height: 1.55;
    }
 
    /* Input group headings */
    .group-title {
        font-weight: 700;
        font-size: 1.02rem;
        color: #0f3d3e;
        margin-bottom: 0.2rem;
    }
    .group-sub {
        font-size: 0.85rem;
        color: #5f7776;
        margin-bottom: 0.8rem;
    }
 
    /* Bordered containers act as cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-radius: 14px;
    }
 
    /* Metric */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #dfe9e7;
        padding: 14px 18px;
        border-radius: 12px;
    }
    div[data-testid="stMetricValue"] {
        color: #0f3d3e;
        font-weight: 800;
    }
 
    /* Button */
    .stButton > button {
        background: #0f3d3e;
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.75rem 1.2rem;
        transition: background 0.2s ease;
    }
    .stButton > button:hover {
        background: #14706f;
        color: #ffffff;
    }
    .stButton > button:focus-visible {
        outline: 3px solid #7fd1c7;
        outline-offset: 2px;
    }
 
    .disclaimer {
        text-align: center;
        color: #6b8281;
        font-size: 0.82rem;
        margin-top: 1.5rem;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
 
# ── 3. Helpers ───────────────────────────────────────────────────────────────
def api_is_online() -> bool:
    """Quick health check on the FastAPI root route."""
    try:
        return requests.get(f"{API_URL}/", timeout=2).status_code == 200
    except requests.exceptions.RequestException:
        return False
 
 
# ── 4. Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        **Model:** K-Nearest Neighbours (k = 5) with StandardScaler  
        **Data:** Pima Indians Diabetes Dataset (768 patients)  
        **Backend:** FastAPI  
        **Frontend:** Streamlit
        """
    )
    st.divider()
    st.subheader("API status")
    if api_is_online():
        st.success("FastAPI is online")
    else:
        st.error("FastAPI is offline")
        st.code("python -m uvicorn main:app --reload", language="bash")
    st.caption(f"Endpoint: {API_URL}/predict")
 
 
# ── 5. Header ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>Diabetes Risk Predictor</h1>
        <p>Enter a patient's clinical measurements. The app sends them to a
        machine learning API and returns a risk prediction with the model's
        confidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
 
tab_predict, tab_about = st.tabs(["Predict", "Feature guide"])
 
# ── 6. Predict tab ───────────────────────────────────────────────────────────
with tab_predict:
    col_a, col_b, col_c = st.columns(3, gap="medium")
 
    with col_a:
        with st.container(border=True):
            st.markdown('<div class="group-title">Background</div>', unsafe_allow_html=True)
            st.markdown('<div class="group-sub">Personal and family history</div>', unsafe_allow_html=True)
            pregnancies = st.slider("Pregnancies", 0, 17, 1)
            age = st.slider("Age (years)", 18, 90, 30)
            dpf = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                max_value=2.5,
                value=0.5,
                step=0.01,
                format="%.3f",
                help="Score based on family history of diabetes.",
            )
 
    with col_b:
        with st.container(border=True):
            st.markdown('<div class="group-title">Blood work</div>', unsafe_allow_html=True)
            st.markdown('<div class="group-sub">Glucose, insulin and pressure</div>', unsafe_allow_html=True)
            glucose = st.slider("Glucose (mg/dL)", 0, 200, 110)
            insulin = st.slider("Insulin (mu U/ml)", 0, 850, 80)
            blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 130, 70)
 
    with col_c:
        with st.container(border=True):
            st.markdown('<div class="group-title">Body measurements</div>', unsafe_allow_html=True)
            st.markdown('<div class="group-sub">Weight and skin fold</div>', unsafe_allow_html=True)
            bmi = st.slider("BMI", 0.0, 70.0, 25.0, step=0.1, format="%.1f")
            skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
 
    # In this dataset, 0 in these fields means "not measured".
    zero_fields = [
        name
        for name, value in [
            ("Glucose", glucose),
            ("Blood Pressure", blood_pressure),
            ("Skin Thickness", skin_thickness),
            ("Insulin", insulin),
            ("BMI", bmi),
        ]
        if value == 0
    ]
    if zero_fields:
        st.warning(
            "These fields are set to 0: "
            + ", ".join(zero_fields)
            + ". A real patient can't have a 0 here, so the prediction may be unreliable."
        )
 
    st.write("")
    predict_clicked = st.button("Predict diabetes risk", use_container_width=True)
 
    if predict_clicked:
        # Field names must match the Pydantic model in main.py exactly.
        payload = {
            "Pregnancies": int(pregnancies),
            "Glucose": float(glucose),
            "BloodPressure": float(blood_pressure),
            "SkinThickness": float(skin_thickness),
            "Insulin": float(insulin),
            "BMI": float(bmi),
            "DiabetesPedigreeFunction": float(dpf),
            "Age": int(age),
        }
 
        friendly_error = (
            "Could not get a prediction. Start the FastAPI server first with "
            "`python -m uvicorn main:app --reload`, then try again."
        )
 
        try:
            with st.spinner("Contacting the prediction API..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
 
            if response.status_code != 200:
                st.error(friendly_error)
            else:
                data = response.json()
                # Supports both "result_label" and "result" as the key name.
                result_label = data.get("result_label") or data.get("result", "Unknown")
                confidence = float(data.get("confidence", 0.0)) * 100
 
                st.write("")
                st.subheader("Prediction report")
                res_left, res_right = st.columns([2, 1], gap="medium")
 
                with res_left:
                    if result_label == "Diabetic":
                        st.error("**Diabetic**: the model predicts a high risk of diabetes.")
                        st.caption("This is a statistical estimate. Please consult a doctor for a real diagnosis.")
                    else:
                        st.success("**Not Diabetic**: the model predicts a low risk of diabetes.")
                        st.caption("A low-risk result does not replace regular health check-ups.")
 
                with res_right:
                    st.metric(label="Model confidence", value=f"{confidence:.1f}%")
                    st.progress(min(max(confidence / 100, 0.0), 1.0))
 
                with st.expander("See the data that was sent to the API"):
                    st.json(payload)
 
        except (requests.exceptions.RequestException, ValueError):
            st.error(friendly_error)
 
# ── 7. Feature guide tab ─────────────────────────────────────────────────────
with tab_about:
    st.subheader("What each input means")
    st.markdown(
        """
        | Feature | Meaning | Range in app |
        |---|---|---|
        | **Pregnancies** | Number of times pregnant | 0 - 17 |
        | **Glucose** | Plasma glucose, 2 hours into an oral glucose tolerance test | 0 - 200 |
        | **Blood Pressure** | Diastolic blood pressure (mm Hg) | 0 - 130 |
        | **Skin Thickness** | Triceps skin fold thickness (mm) | 0 - 100 |
        | **Insulin** | 2-hour serum insulin (mu U/ml) | 0 - 850 |
        | **BMI** | Weight in kg / (height in m)² | 0 - 70 |
        | **Diabetes Pedigree Function** | Diabetes likelihood score from family history | 0.0 - 2.5 |
        | **Age** | Age in years | 18 - 90 |
        """
    )
    st.info(
        "How it works: Streamlit sends these 8 values as JSON to FastAPI. "
        "FastAPI scales them, runs the KNN model, and returns the prediction and confidence."
    )
 
# ── 8. Footer ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="disclaimer">
        Educational portfolio project. It is not a medical device and must not
        be used for real diagnosis.<br>
        Built with FastAPI, Streamlit and scikit-learn.
    </div>
    """,
    unsafe_allow_html=True,
)
 