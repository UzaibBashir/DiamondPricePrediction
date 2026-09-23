"""Streamlit interface for the trained diamond price model."""

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.diamond_price_prediction.preprocessing import create_features, encode_ordinals


MODEL_PATH = ROOT / "model" / "diamond_price_pipeline.pkl"

st.set_page_config(page_title="Diamond Price Estimator", page_icon="💎", layout="centered")
st.title("Diamond Price Estimator")
st.caption("Educational estimate based on the project's public diamond dataset.")

if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python train_model.py` from the project root first.")
    st.stop()

with st.form("diamond_inputs"):
    st.subheader("Diamond characteristics")
    carat = st.number_input("Carat", min_value=0.1, max_value=5.5, value=1.0, step=0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    color = st.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"], index=4)
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=3)
    depth = st.number_input("Depth (%)", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
    table = st.number_input("Table (%)", min_value=40.0, max_value=80.0, value=57.0, step=0.1)
    dimensions = st.columns(3)
    x = dimensions[0].number_input("Length x (mm)", min_value=0.1, max_value=15.0, value=6.4, step=0.01)
    y = dimensions[1].number_input("Width y (mm)", min_value=0.1, max_value=15.0, value=6.4, step=0.01)
    z = dimensions[2].number_input("Depth z (mm)", min_value=0.1, max_value=15.0, value=3.9, step=0.01)
    submitted = st.form_submit_button("Estimate price")

if submitted:
    row = pd.DataFrame([{"carat": carat, "cut": cut, "color": color, "clarity": clarity,
                         "depth": depth, "table": table, "x": x, "y": y, "z": z}])
    row = create_features(encode_ordinals(row))
    model = joblib.load(MODEL_PATH)
    estimate = max(0.0, float(model.predict(row)[0]))
    st.metric("Estimated price", f"${estimate:,.0f}")
    st.info("This is a model estimate, not a professional appraisal or guaranteed selling price.")