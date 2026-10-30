import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load saved model files
@st.cache_resource
def load_model_files():
    model = tf.keras.models.load_model("loan_approval_ann.keras")
    preprocessor = joblib.load("loan_approval_preprocessor.pkl")
    label_encoder = joblib.load("loan_approval_label_encoder.pkl")
    return model, preprocessor, label_encoder

try:
    model, preprocessor, label_encoder = load_model_files()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Application heading
st.title("🏦 Loan Approval Prediction System")

st.write(
    "This academic project uses an Artificial Neural Network (ANN) "
    "to demonstrate loan approval prediction."
)

st.info(
    "Enter the applicant information below and click Predict "
    "to see the model's prediction."
)

# Applicant input form
st.header("Applicant Information")

with st.form("loan_prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=20,
            value=2,
            step=1
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        income_annum = st.number_input(
            "Annual Income",
            min_value=0,
            value=5000000,
            step=100000
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=15000000,
            step=100000
        )

        loan_term = st.number_input(
            "Loan Term (Years)",
            min_value=1,
            max_value=50,
            value=10
        )

    with col2:
        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=0,
            max_value=1000,
            value=700
        )

        residential_assets_value = st.number_input(
            "Residential Assets Value",
            min_value=0,
            value=5000000,
            step=100000
        )

        commercial_assets_value = st.number_input(
            "Commercial Assets Value",
            min_value=0,
            value=3000000,
            step=100000
        )

        luxury_assets_value = st.number_input(
            "Luxury Assets Value",
            min_value=0,
            value=10000000,
            step=100000
        )

        bank_asset_value = st.number_input(
            "Bank Asset Value",
            min_value=0,
            value=4000000,
            step=100000
        )

    predict_button = st.form_submit_button(
        "Predict Loan Approval",
        use_container_width=True
    )

# Prediction
if predict_button:

    input_data = pd.DataFrame([{
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    try:
        processed_data = preprocessor.transform(input_data)

        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()

        processed_data = np.asarray(processed_data, dtype=np.float32)

        prediction_probability = model.predict(
            processed_data,
            verbose=0
        )[0][0]

        predicted_class = (
            1 if prediction_probability >= 0.5 else 0
        )

        prediction_label = label_encoder.inverse_transform(
            np.array([predicted_class])
        )[0]

        st.subheader("Prediction Result")

      prediction_label = str(prediction_label).strip()

if prediction_label == "Approved":
    st.success(f"Prediction: {prediction_label}")
else:
    st.warning(f"Prediction: {prediction_label}")

        st.write(
            f"Model output probability: {prediction_probability:.4f}"
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# Model information
st.divider()

st.header("About the Model")

col1, col2 = st.columns(2)

with col1:
    st.metric("ANN Test Accuracy", "97.31%")

with col2:
    st.write("**Model:** Artificial Neural Network (ANN)")
    st.write("**Framework:** TensorFlow / Keras")

st.caption(
    "Academic demonstration only. Predictions are not actual "
    "bank lending decisions or financial advice."
)
