import streamlit as st
import sys
import os

# --- Path Setup ---
# This finds the absolute path to the 'app' folder, then goes one level up to the root
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

# Add the root directory to sys.path so it can see 'src'
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Now we can safely import
try:
    from src.predict import make_prediction
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

st.set_page_config(page_title="Customer Retention AI", layout="wide")

st.title("📡 Telco Customer Churn Risk Analysis")
st.markdown("---")

# Layout Columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Profile")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.radio("Paperless Billing", ["Yes", "No"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = tenure * monthly_charges # Helper logic

with col2:
    st.subheader("🌐 Services & Support")
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

# Button to predict
if st.button("Analyze Retention Risk", use_container_width=True):
    # Prepare dummy structure for one-hot encoded columns 
    # (Note: In a production app, you'd automate the encoding mapping)
    # For this version, ensure your mapping matches the X_train columns!
    
    # Simple logic to demonstrate the dashboard
    input_features = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        # ... other features mapped to match model columns ...
    }
    
    # Placeholder for our prediction logic
    # In a full SDLC, you'd ensure your input_features matches X_train exactly
    st.markdown("### 📊 Prediction Results")
    
    # Mock probability for visual demonstration
    risk_prob = 0.85 if contract == "Month-to-month" and tenure < 6 else 0.15
    
    if risk_prob > 0.5:
        st.error(f"🔴 **HIGH RISK:** This customer has a {risk_prob*100:.1f}% chance of churning.")
        st.progress(risk_prob)
        st.warning("👉 **Recommendation:** Offer a 1-year contract discount or a loyalty tech support bundle.")
    else:
        st.success(f"🟢 **LOW RISK:** Retention probability is high ({ (1-risk_prob)*100:.1f}%).")
        st.progress(risk_prob)