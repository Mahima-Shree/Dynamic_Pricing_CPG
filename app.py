import streamlit as st
import pickle
import numpy as np

# Load the trained XGBoost model
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🛒 Tide Dynamic Pricing Predictor")
st.markdown("### Enter competitor pricing and discount details:")

# Collect user input
input_features = []
feature_names = ['NoPromoPrice', 'DiscountImpact', 'Revenue_With_Discount', 'Revenue_Without_Discount', 'UnitsSold']

for feature in feature_names:
    value = st.number_input(f"{feature}:", min_value=0.0, format="%.2f")
    input_features.append(value)

# Prediction button
if st.button("💰 Predict Selling Price"):
    features_array = np.array(input_features).reshape(1, -1)
    predicted_price = model.predict(features_array)[0]
    st.success(f"🔮 Predicted Selling Price: **${predicted_price:.2f}**")

# Footer
st.markdown("---")
st.markdown("💡 This model helps optimize pricing based on competitor data and market trends.")
