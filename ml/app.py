import subprocess
import string
import re
import streamlit as st
import joblib
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import pickle

# 設定路徑
MODELS_DIR = Path("models")
ARTIFACTS_DIR = Path("models/artifacts")

# 確保目錄存在
MODELS_DIR.mkdir(exist_ok=True)
ARTIFACTS_DIR.mkdir(exist_ok=True)

# 資料集 URL
DATASET_URL = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"

st.set_page_config(
    page_title="Spam Classifier Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自訂 CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stCheckbox {
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 資料載入 ====================
@st.cache_data
def load_data():
    """載入資料集"""
    try:
        df = pd.read_csv(DATASET_URL, names=['label', 'text'], encoding='utf-8')
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None

# ==================== 模型載入 ====================
@st.cache_resource
def load_model_artifacts():
    """載入模型相關檔案"""
    model_path = MODELS_DIR / "spam_classifier.pkl"
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"
    label_encoder_path = MODELS_DIR / "label_encoder.pkl"
    
    # 檢查檔案是否存在
    missing_files = []
    if not model_path.exists():
        missing_files.append("spam_classifier.pkl")
    if not vectorizer_path.exists():
        missing_files.append("vectorizer.pkl")
    if not label_encoder_path.exists():
        missing_files.append("label_encoder.pkl")
    
    if missing_files:
        st.warning(f"⚠️ Missing files: {', '.join(missing_files)}")
        st.info("💡 Please train the model first by running: `python train_baseline.py`")
        return None, None, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        return model, vectorizer, label_encoder
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 Try retraining the model: `python train_baseline.py`")
        return None, None, None


# ==================== 載入 Metrics ====================
def load_metrics():
    """載入訓練指標"""
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

# ==================== 文字預處理 ====================
def preprocess_text(text, lowercase=True, remove_punct=True, remove_stopwords=False):
    """預處理文字"""
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", text)
    if remove_stopwords:
        tokens = [t for t in text.split() if t not in ENGLISH_STOP_WORDS]
        text = " ".join(tokens)
    return text

# ==================== 視覺化函數 ====================
def get_top_tokens(df, label, n_tokens=20):
    """取得 Top N tokens"""
    texts = df[df['label'] == label]['text'].values
    all_text = ' '.join(texts)
    tokens = re.findall(r'\b\w+\b', all_text.lower())
    token_counts = Counter(tokens)
    return token_counts.most_common(n_tokens)

def plot_top_tokens(df, n_tokens=20):
    """繪製 Top Tokens 對比圖"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # HAM tokens
    ham_tokens = get_top_tokens(df, 'ham', n_tokens)
    ham_words, ham_counts = zip(*ham_tokens)
    colors_ham = plt.cm.viridis(np.linspace(0.3, 0.9, len(ham_words)))
    ax1.barh(range(len(ham_words)), ham_counts, color=colors_ham)
    ax1.set_yticks(range(len(ham_words)))
    ax1.set_yticklabels(ham_words)
    ax1.invert_yaxis()
    ax1.set_xlabel('Frequency', fontsize=12)
    ax1.set_ylabel('Token', fontsize=12)
    ax1.set_title('Class: ham', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # SPAM tokens
    spam_tokens = get_top_tokens(df, 'spam', n_tokens)
    spam_words, spam_counts = zip(*spam_tokens)
    colors_spam = plt.cm.viridis(np.linspace(0.3, 0.9, len(spam_words)))
    ax2.barh(range(len(spam_words)), spam_counts, color=colors_spam)
    ax2.set_yticks(range(len(spam_words)))
    ax2.set_yticklabels(spam_words)
    ax2.invert_yaxis()
    ax2.set_xlabel('Frequency', fontsize=12)
    ax2.set_ylabel('Token', fontsize=12)
    ax2.set_title('Class: spam', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle(f'Top Tokens by Class 📊', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig

def plot_class_distribution(df):
    """繪製類別分布圖"""
    fig, ax = plt.subplots(figsize=(8, 6))
    class_counts = df['label'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax.pie(
        class_counts.values, 
        labels=class_counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )
    ax.set_title('Class Distribution 🥧', fontsize=14, fontweight='bold', pad=20)
    return fig

def plot_message_length_distribution(df):
    """繪製訊息長度分布"""
    df['message_length'] = df['text'].str.len()
    fig, ax = plt.subplots(figsize=(10, 6))
    ham_lengths = df[df['label'] == 'ham']['message_length']
    spam_lengths = df[df['label'] == 'spam']['message_length']
    ax.hist(ham_lengths, bins=50, alpha=0.6, label='HAM', color='#2ecc71', edgecolor='black')
    ax.hist(spam_lengths, bins=50, alpha=0.6, label='SPAM', color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Message Length (characters)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Message Length Distribution 📏', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    return fig

# ==================== 主程式 ====================
st.markdown('<h1 class="main-header">🚀 Spam Email/SMS Classification Demo</h1>', unsafe_allow_html=True)

# 載入資料
df = load_data()

if df is not None:
    st.success(f"✅ Dataset loaded: {len(df)} messages")
    
    # ==================== 側邊欄 ====================
    with st.sidebar:
        st.header("📊 Configuration")
        
        # 模式選擇
        st.subheader("Mode Selection")
        mode = st.radio("Choose mode", ["Single message", "Batch upload (CSV)"], label_visibility="collapsed")
        
        st.divider()
        
        # 模型參數
        st.subheader("⚙️ Model Parameters")
        model_choice = st.selectbox("Model", ["Logistic Regression", "SVM"])
        confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
        
        st.divider()
        
        # 文字預處理選項
        st.subheader("🔧 Text Preprocessing")
        opt_lowercase = st.toggle("Lowercase", value=True)
        opt_remove_punct = st.toggle("Remove punctuation", value=True)
        opt_remove_stopwords = st.toggle("Remove stopwords", value=False)
        
        st.divider()
        
        # 顯示選項
        st.subheader("👁️ Display Options")
        opt_show_preproc = st.toggle("Show preprocessing steps", value=True)
        opt_show_probs = st.toggle("Show probability scores", value=True)
        opt_show_cm = st.toggle("Show confusion matrix", value=True)
        opt_show_roc = st.toggle("Show ROC curve", value=True)
        
        st.divider()
        
        # Data Overview 控制
        st.subheader("📊 Data Overview")
        show_raw_data = st.toggle("Show raw data", value=True)
        n_samples = st.slider("Number of samples", 5, 50, 10)
        
        st.divider()
        
        # 圖表控制
        st.subheader("📈 Visualizations")
        n_tokens = st.slider("Top N tokens", 10, 50, 20, 5)
        show_class_dist = st.toggle("Show class distribution", value=True)
        show_length_dist = st.toggle("Show message length", value=True)
        show_token_chart = st.toggle("Show top tokens", value=True)
        
        st.divider()
        
        # 重新訓練
        st.subheader("🔄 Retrain Model")
        with st.expander("Training Settings"):
            retrain_max_features = st.number_input("Max features", 100, 50000, 5000, 100)
            retrain_C = st.number_input("Regularization C", 0.0001, 100.0, 1.0)
            retrain_test_size = st.slider("Test size", 0.05, 0.5, 0.2, 0.01)
            retrain_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
    
    # ==================== 主要內容區 ====================
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "🔮 Prediction", "📈 Visualizations", "📋 Model Info"])
    
    # ==================== Tab 1: Data Overview ====================
    with tab1:
        st.header("📊 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Messages", len(df))
        with col2:
            st.metric("HAM Messages", len(df[df['label'] == 'ham']))
        with col3:
            st.metric("SPAM Messages", len(df[df['label'] == 'spam']))
        with col4:
            spam_ratio = len(df[df['label'] == 'spam']) / len(df) * 100
            st.metric("SPAM Ratio", f"{spam_ratio:.1f}%")
        
        st.divider()
        
        if show_raw_data:
            st.subheader(f"📄 Dataset Preview (First {n_samples} rows)")
            st.dataframe(df.head(n_samples), use_container_width=True)
        
        st.subheader("📊 Detailed Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**HAM Statistics:**")
            ham_df = df[df['label'] == 'ham']
            ham_lengths = ham_df['text'].str.len()
            st.write(f"- Count: {len(ham_df)}")
            st.write(f"- Avg length: {ham_lengths.mean():.1f} chars")
            st.write(f"- Min length: {ham_lengths.min()} chars")
            st.write(f"- Max length: {ham_lengths.max()} chars")
        with col2:
            st.write("**SPAM Statistics:**")
            spam_df = df[df['label'] == 'spam']
            spam_lengths = spam_df['text'].str.len()
            st.write(f"- Count: {len(spam_df)}")
            st.write(f"- Avg length: {spam_lengths.mean():.1f} chars")
            st.write(f"- Min length: {spam_lengths.min()} chars")
            st.write(f"- Max length: {spam_lengths.max()} chars")
    
    # ==================== Tab 2: Prediction ====================
    with tab2:
        st.header("🔮 Message Classification")
        
        model, vectorizer, label_encoder = load_model_artifacts()
        
        if model is None:
            st.error("❌ No model found. Train the model from the sidebar.")
        else:
            st.success("✅ Model loaded successfully!")
            
            if mode == "Single message":
                st.subheader("Enter message text")
                input_text = st.text_area("", placeholder="Type your message here...", height=150, value="Free entry for demonstration")
                
                if st.button("Classify", type="primary"):
                    if input_text.strip():
                        try:
                            # 預處理
                            x_proc = preprocess_text(input_text, opt_lowercase, opt_remove_punct, opt_remove_stopwords)
                            
                            # 向量化
                            text_vector = vectorizer.transform([x_proc])
                            
                            # 預測
                            prediction = model.predict(text_vector)[0]
                            proba = model.predict_proba(text_vector)[0]
                            
                            # 解碼標籤
                            label = label_encoder.inverse_transform([prediction])[0]
                            confidence = max(proba) * 100
                            spam_prob = proba[1]
                            
                            # 顯示結果
                            if label == 'spam':
                                st.error(f"⚠️ **SPAM** detected with {confidence:.1f}% confidence")
                            else:
                                st.success(f"✅ **HAM** (legitimate) with {confidence:.1f}% confidence")
                            
                            # 信心度檢查
                            if spam_prob >= confidence_threshold:
                                st.warning(f"⚠️ Confidence threshold ({confidence_threshold:.2f}) — Prediction ACCEPTED")
                            else:
                                st.info(f"ℹ️ Confidence threshold ({confidence_threshold:.2f}) — Prediction REJECTED")
                            
                            # 顯示機率
                            if opt_show_probs:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("HAM Probability", f"{proba[0]*100:.1f}%")
                                with col2:
                                    st.metric("SPAM Probability", f"{proba[1]*100:.1f}%")
                            
                            # 顯示預處理步驟
                            if opt_show_preproc:
                                st.subheader("🔍 Preprocessing Steps")
                                st.write("**Original:**", input_text)
                                st.write("**Processed:**", x_proc)
                                st.write("**Tokens:**", x_proc.split())
                                
                                # Top TF-IDF features
                                try:
                                    feature_names = vectorizer.get_feature_names_out()
                                    x_vec = text_vector.toarray()[0]
                                    top_idx = np.argsort(x_vec)[-10:][::-1]
                                    top_features = [(feature_names[i], float(x_vec[i])) for i in top_idx if x_vec[i] > 0]
                                    if top_features:
                                        st.write("**Top TF-IDF features:**")
                                        st.table(pd.DataFrame(top_features, columns=['Feature', 'TF-IDF Score']))
                                except Exception:
                                    pass
                                
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")
                    else:
                        st.warning("⚠️ Please enter a message")
            
            else:  # Batch upload
                st.subheader("Upload CSV file")
                uploaded_file = st.file_uploader("Choose a CSV file", type=['csv', 'txt'])
                
                if uploaded_file is not None:
                    try:
                        uploaded_df = pd.read_csv(uploaded_file, header=None)
                        if uploaded_df.shape[1] >= 2:
                            texts = uploaded_df.iloc[:, 1].astype(str).tolist()
                        else:
                            texts = uploaded_df.iloc[:, 0].astype(str).tolist()
                        
                        st.write("**Preview:**")
                        st.dataframe(uploaded_df.head())
                        
                        if st.button("Classify All", type="primary"):
                            with st.spinner("Processing..."):
                                # 預處理
                                texts_proc = [preprocess_text(t, opt_lowercase, opt_remove_punct, opt_remove_stopwords) for t in texts]
                                
                                # 向量化
                                text_vectors = vectorizer.transform(texts_proc)
                                
                                # 預測
                                predictions = model.predict(text_vectors)
                                probas = model.predict_proba(text_vectors)
                                
                                # 解碼
                                labels = label_encoder.inverse_transform(predictions)
                                confidences = np.max(probas, axis=1) * 100
                                spam_probs = probas[:, 1]
                                
                                # 建立結果 DataFrame
                                result_df = pd.DataFrame({
                                    'text': texts,
                                    'processed': texts_proc,
                                    'prediction': labels,
                                    'spam_prob': spam_probs,
                                    'confidence': confidences
                                })
                                
                                st.success("✅ Classification completed!")
                                st.dataframe(result_df, use_container_width=True)
                                
                                # 下載結果
                                csv = result_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download results",
                                    data=csv,
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                )
                    except Exception as e:
                        st.error(f"❌ Error processing file: {e}")
    
    # ==================== Tab 3: Visualizations ====================
    with tab3:
        st.header("📈 Data Visualizations")
        
        if show_token_chart:
            with st.spinner("Generating top tokens chart..."):
                fig = plot_top_tokens(df, n_tokens)
                st.pyplot(fig, use_container_width=True)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        if show_class_dist:
            with col1:
                with st.spinner("Generating class distribution..."):
                    fig = plot_class_distribution(df)
                    st.pyplot(fig, use_container_width=True)
        
        if show_length_dist:
            with col2:
                with st.spinner("Generating length distribution..."):
                    fig = plot_message_length_distribution(df)
                    st.pyplot(fig, use_container_width=True)
        
        # 顯示混淆矩陣和 ROC 曲線（如果存在）
        st.divider()
        st.subheader("🎯 Model Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if opt_show_cm:
                cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
                if cm_path.exists():
                    st.image(str(cm_path), caption="Confusion Matrix")
                else:
                    st.info("ℹ️ Train model to generate confusion matrix")
        
        with col2:
            if opt_show_roc:
                roc_path = ARTIFACTS_DIR / "roc_curve.png"
                if roc_path.exists():
                    st.image(str(roc_path), caption="ROC Curve")
                else:
                    st.info("ℹ️ Train model to generate ROC curve")
    
    # ==================== Tab 4: Model Info ====================
    with tab4:
        st.header("📋 Model Information")
        
        model, vectorizer, label_encoder = load_model_artifacts()
        
        if model is not None:
            st.success("✅ Model loaded successfully")
            
            st.subheader("Model Parameters")
            st.json({
                "Model Type": type(model).__name__,
                "Vectorizer": type(vectorizer).__name__,
                "Max Features": vectorizer.max_features if hasattr(vectorizer, 'max_features') else "N/A",
                "Vocabulary Size": len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else "N/A"
            })
            
            metrics = load_metrics()
            if metrics is not None:
                st.subheader("📊 Model Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                with col2:
                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                    st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
                with col3:
                    st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        else:
            st.warning("⚠️ No model found. Please train a model first.")
    
    # ==================== 重新訓練處理 ====================
    if retrain_button:
        st.sidebar.info("⏳ Training in progress...")
        # 這裡需要你的訓練函數
        st.sidebar.warning("⚠️ Training function not implemented. Please use train_baseline.py manually.")

else:
    st.error("❌ Failed to load dataset")
