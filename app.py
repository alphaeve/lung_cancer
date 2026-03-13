import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Lung Cancer Risk AI", layout="wide")

model = joblib.load("model.pkl")

st.title("AI System for Early Lung Cancer Risk Prediction")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", [0,1])  # 0=Female, 1=Male
    age = st.slider("Age", 20, 80, 40)
    smoking = st.selectbox("Smoking", [0,1])
    yellow_fingers = st.selectbox("Yellow Fingers", [0,1])
    anxiety = st.selectbox("Anxiety", [0,1])
    peer_pressure = st.selectbox("Peer Pressure", [0,1])
    chronic_disease = st.selectbox("Chronic Disease", [0,1])

with col2:
    fatigue = st.selectbox("Fatigue", [0,1])
    allergy = st.selectbox("Allergy", [0,1])
    wheezing = st.selectbox("Wheezing", [0,1])
    alcohol = st.selectbox("Alcohol Consuming", [0,1])
    coughing = st.selectbox("Coughing", [0,1])
    breath = st.selectbox("Shortness of Breath", [0,1])
    swallow = st.selectbox("Swallowing Difficulty", [0,1])
    chest_pain = st.selectbox("Chest Pain", [0,1])

data = np.array([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                  chronic_disease, fatigue, allergy, wheezing, alcohol,
                  coughing, breath, swallow, chest_pain]])

if st.button("Predict Lung Cancer Risk"):
    pred = model.predict(data)[0]

    if pred == 1:
        st.error("High Lung Cancer Risk")
    else:
        st.success("Low Lung Cancer Risk")

st.markdown("---")
st.subheader("Explainable AI")
st.image("shap_feature_importance.png")








# import streamlit as st
# import numpy as np
# import joblib
# from pollution_api import get_pollution

# st.set_page_config(page_title="Lung Cancer Risk AI",layout="wide")

# model = joblib.load("model.pkl")

# st.title("AI System for Early Lung Cancer Risk Prediction")

# col1,col2 = st.columns(2)

# with col1:
#     age = st.slider("Age",20,80,40)
#     bmi = st.slider("BMI",15,40,25)
#     smoking = st.selectbox("Smoking",[0,1])
#     alcohol = st.selectbox("Alcohol Use",[0,1])
#     biomass = st.selectbox("Biomass Fuel Exposure",[0,1])
#     copd = st.selectbox("COPD History",[0,1])

# with col2:
#     lifestyle = st.slider("Lifestyle Risk Score",1,10,5)
#     district = st.slider("District Pollution Index",20,120,60)
#     pm25 = get_pollution()

# st.write("Live Pollution PM2.5 (API):",pm25)

# data = np.array([[age,bmi,smoking,alcohol,pm25,biomass,copd,district,lifestyle]])

# if st.button("Predict Lung Cancer Risk"):
#     pred = model.predict(data)[0]

#     if pred == 1:
#         st.error("High Lung Cancer Risk")
#     else:
#         st.success("Low Lung Cancer Risk")

# st.markdown("---")
# st.subheader("Explainable AI")
# st.image("shap_feature_importance.png")
