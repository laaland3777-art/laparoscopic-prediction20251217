import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 页面配置 ---
st.set_page_config(
    page_title="Laparoscopic Surgery Difficulty Prediction",
    layout="centered"
)

# --- 加载模型和标准化器 ---
@st.cache_resource
def load_model_and_scaler():
    try:
        # 请确保 GitHub 上的文件名是 'ensemble_model.pkl' 和 'scaler.pkl'
        # 如果你之前上传的是 xgboost_model.pkl，请在这里改名
        model = joblib.load('ensemble_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'ensemble_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_model_and_scaler()

# --- 标题和介绍 ---
st.title("Laparoscopic Surgery Difficulty Prediction Model")
st.markdown("""
This application predicts the difficulty probability of laparoscopic surgery based on preoperative clinical features.
Please input the patient's parameters below.
""")

st.markdown("---")

# --- 侧边栏或主界面的输入表单 ---
st.subheader("Patient Features Input")

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    # 1. Abdominal wall adipose area
    f1 = st.number_input(
        "Abdominal wall adipose area (mm²)", 
        min_value=0.0, 
        value=100.0, 
        step=1.0,
        help="Area of subcutaneous fat in the abdominal wall."
    )
    
    # 2. SMV adipose thickness
    f2 = st.number_input(
        "SMV adipose thickness (mm)", 
        min_value=0.0, 
        value=10.0, 
        step=0.1,
        help="Thickness of adipose tissue around the Superior Mesenteric Vein."
    )
    
    # 3. SMV adipose density
    f3 = st.number_input(
        "SMV adipose density (HU)", 
        value=-80.0, 
        step=1.0,
        help="Radiodensity of SMV fat in Hounsfield Units."
    )
    
    # 4. Type of Henle's trunk (罗马数字 I-IV)
    henle_map = {"I": 0, "II": 1, "III": 2, "IV": 3}
    
    f4_display = st.selectbox(
        "Type of Henle's trunk",
        options=["I", "II", "III", "IV"], 
        index=0,
        help="Anatomical variation type of the Gastrocolic trunk of Henle."
    )
    f4 = henle_map[f4_display]

with col2:
    # 5. Intra-abdominal adipose area
    f5 = st.number_input(
        "Intra-abdominal adipose area (mm²)", 
        min_value=0.0, 
        value=150.0, 
        step=1.0,
        help="Visceral fat area."
    )
    
    # 6. Presence of the right colonic artery (修改点：去掉了 (0) 和 (1))
    f6_display = st.selectbox(
        "Presence of the right colonic artery",
        options=["Absent", "Present"], # 这里只显示单词，更美观
        index=0
    )
    # 转换逻辑：如果是 Present 则为 1，否则为 0
    f6 = 1 if f6_display == "Present" else 0
    
    # 7. Plasma triglycerides (Normal/High)
    f7_display = st.selectbox(
        "Plasma triglycerides",
        options=["Normal", "High"], 
        index=0,
        help="Normal level vs High level"
    )
    # 转换逻辑：Normal -> 0, High -> 1
    f7 = 0 if f7_display == "Normal" else 1

# --- 预测逻辑 ---
if st.button("Predict Difficulty", type="primary"):
    if model is not None and scaler is not None:
        # 1. 构造输入 DataFrame (必须与训练时的列名一致)
        feature_names = [
            'Abdominal wall adipose area',
            'SMV adipose thickness',
            'SMV adipose density',
            "Type of Henle's trunk",
            'Intra-abdominal adipose area',
            'Presence of the right colonic artery',
            'Plasma triglycerides'
        ]
        
        # 这里的 f4, f6, f7 已经是转换好的数字了
        input_data = pd.DataFrame([[f1, f2, f3, f4, f5, f6, f7]], columns=feature_names)
        
        # 2. 标准化
        input_scaled = scaler.transform(input_data)
        
        # 3. 预测概率
        probability = model.predict_proba(input_scaled)[0][1]
        prediction_class = 1 if probability >= 0.5 else 0
        
        # --- 显示结果 ---
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # 进度条
        st.progress(probability)
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric(label="Difficulty Probability", value=f"{probability:.2%}")
            
        with result_col2:
            if prediction_class == 1:
                st.error("High Difficulty Predicted")
            else:
                st.success("Low Difficulty Predicted")
                
        st.info(f"The model predicts a **{probability:.1%}** chance of the surgery being difficult.")

# --- 页脚 ---
st.markdown("---")
st.caption("Model based on Weighted Ensemble (RF, SVM, LR, GNB, AdaBoost).")

