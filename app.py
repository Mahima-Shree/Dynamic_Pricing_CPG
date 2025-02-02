import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🛒 Tide Dynamic Pricing Predictor")
st.markdown("### Enter competitor pricing and discount details:")

# Define feature names and allow negative values for certain features
feature_names = [
    ('NoPromoPrice', 0.0),  # Only non-negative values
    ('DiscountImpact', None),  # Can be negative
    ('Revenue_With_Discount', None),  # Can be negative
    ('Revenue_Without_Discount', None),  # Can be negative
    ('UnitsSold', 0)  # Only non-negative values
]

input_features = []

for feature, min_val in feature_names:
    if min_val is not None:
        value = st.number_input(f"{feature}:", min_value=min_val, format="%.2f")
    else:
        value = st.number_input(f"{feature}:", format="%.2f")
    input_features.append(value)

# Prediction button
if st.button("💰 Predict Selling Price"):
    features_array = np.array(input_features).reshape(1, -1)
    predicted_price = model.predict(features_array)[0]
    st.success(f"🔮 Predicted Selling Price: **${predicted_price:.2f}**")

# Footer
st.markdown("---")
st.markdown("💡 This model helps optimize pricing based on competitor data and market trends.")
