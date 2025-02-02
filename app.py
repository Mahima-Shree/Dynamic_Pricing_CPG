import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained XGBoost model
with open("xgboost_best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Tide Dynamic Pricing Predictor", page_icon="💰", layout="centered")

st.title("🛒 Tide Dynamic Pricing Predictor")
st.markdown("### 📊 Optimize Your Selling Price Using AI!")

st.markdown(
    "Enter your product pricing details below, and let our AI-powered model predict the best selling price for **maximum revenue**. 💡"
)

# Sidebar with details
st.sidebar.header("📌 About the Model")
st.sidebar.info(
    "This AI model analyzes competitor pricing, discount strategies, and past revenue trends "
    "to provide the most **optimal selling price** for your product."
)

# Define feature names and allow negative values for certain features 
feature_names = [
    ('NoPromoPrice', 0.0),  # Only non-negative values
    ('DiscountRate', None),  # Can be negative
    ('Revenue_With_Discount', None),  # Can be negative
    ('Revenue_Without_Discount', None),  # Can be negative
    ('UnitsSold', 0)  # Only non-negative values
]

input_features = []

# User inputs
st.markdown("### 🔢 Enter Product Details:")
for feature, min_val in feature_names:
    if min_val is not None:
        value = st.number_input(f"🔹 {feature}:", min_value=min_val, format="%.2f")
    else:
        value = st.number_input(f"🔹 {feature}:", format="%.2f")
    input_features.append(value)

# Convert to array
features_array = np.array(input_features).reshape(1, -1)

# Prediction button
if st.button("💰 Predict Selling Price"):
    predicted_price = model.predict(features_array)[0]
    st.success(f"🎯 **Predicted Selling Price:** **${predicted_price:.2f}** 💵")
    
    # Additional insights
    st.markdown("---")
    st.subheader("📉 Key Insights:")
    if predicted_price < input_features[0]:
        st.warning("⚠️ The predicted price is **lower** than your non-promo price. Consider adjusting your strategy.")
    else:
        st.success("✅ The AI suggests **an optimal price increase** for better profitability.")

    # Optional: Feature Importance (if using XGBoost)
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "Feature": [f[0] for f in feature_names],
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.markdown("### 🔍 Feature Importance:")
        st.dataframe(feature_importance)

# Footer
st.markdown("---")
st.markdown("💡 **AI-driven pricing strategies to maximize your revenue!** 🚀")
