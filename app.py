import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import numpy as np
import os
import gdown

st.set_page_config(page_title="MoodVibe", page_icon="😊", layout="wide")

model_path = "./stress_model"
drive_folder_id = "1KpwcYgQcNxwns6sMyl5bBzIGtZtE0kH5"

def download_model_from_drive():
    if not os.path.exists(model_path) or not all(
        os.path.exists(os.path.join(model_path, f)) 
        for f in ["label_encoder.pkl", "tokenizer_config.json", "model.safetensors"]
    ):
        os.makedirs(model_path, exist_ok=True)
        st.info("Downloading model files from Google Drive...")
        try:
            gdown.download_folder(id=drive_folder_id, output=model_path, quiet=False, use_cookies=False)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            raise
    

# Call download function and ensure it completes before loading
download_model_from_drive()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_label_encoder():
    try:
        with open(f"{model_path}/label_encoder.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Label encoder file not found. Ensure model files are correctly downloaded.")
        raise

@st.cache_resource
def load_tokenizer():
    try:
        return BertTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        st.error(f"Failed to load tokenizer: {str(e)}")
        raise

@st.cache_resource
def load_model():
    try:
        model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        raise

# Load resources only after download is confirmed
label_encoder = load_label_encoder()
tokenizer = load_tokenizer()
model = load_model()

# Rest of your code (predict_sentiment, UI, etc.) remains unchanged


def predict_sentiment(post, max_length=128):
    inputs = tokenizer(
        post,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
    return label_encoder.inverse_transform([predicted_class])[0], confidence


st.markdown(
    """
<style>
    .stButton>button {
        background: linear-gradient(45deg, #1DA1F2, #0d8bf2);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 24px;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .stTextArea textarea {
        border: 2px solid #1DA1F2;
        border-radius: 12px;
        font-size: 16px;
        padding: 12px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .history-item {
        background: #f5f5f5;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .progress-bar {
        background: #e0e0e0;
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
    }
    .progress-fill {
        background: #1DA1F2;
        height: 100%;
        transition: width 1s ease-in-out;
    }
    #char-count { font-size: 14px; color: #333333; margin-top: 5px; }
    #token-count { font-size: 14px; color: #333333; margin-left: 10px; }
    .tooltip { position: relative; display: inline-block; cursor: help; }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
<script>
    function updateCharCount() {
        const textareas = document.querySelectorAll('textarea');
        textareas.forEach(textarea => {
            const charCount = textarea.closest('.stTextArea').querySelector('#char-count');
            if (charCount) {
                charCount.textContent = `${textarea.value.length}/${char_limit}`;
                charCount.style.color = textarea.value.length > ${char_limit} ? '#ff5555' : '#333333';
            }
        });
    }
    document.addEventListener('DOMContentLoaded', () => {
        const textareas = document.querySelectorAll('textarea');
        textareas.forEach(textarea => {
            textarea.addEventListener('input', updateCharCount);
        });
        updateCharCount();
    });
    const observer = new MutationObserver(updateCharCount);
    observer.observe(document.body, { childList: true, subtree: true });
</script>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    max_length = st.slider(
        "Max Tokens for Analysis",
        32,
        512,
        128,
        8,
        help="Sets the maximum number of tokens (words or subwords) the model processes. Higher values capture more text but increase computation time.",
    )
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    debug_mode = st.sidebar.checkbox("Show Confidence Scores")
    char_limit_enabled = st.sidebar.checkbox("Enable Character Limit", value=True)
    char_limit = (
        st.sidebar.number_input(
            "Character Limit", min_value=100, max_value=10000, value=280, step=10
        )
        if char_limit_enabled
        else float("inf")
    )
    st.markdown("---")
    st.markdown("😊 MoodVibe v1.0")

if "history" not in st.session_state:
    st.session_state.history = []
else:
    for item in st.session_state.history:
        if "status" in item and "sentiment" not in item:
            item["sentiment"] = item.pop("status")

st.title("😊 MoodVibe")
st.markdown(
    "Discover the emotions behind social media posts with AI-driven sentiment analysis."
)

col1, col2 = st.columns([3, 2])
with col1:
    user_input = st.text_area(
        "Enter Post",
        placeholder="Type a post (e.g., 'Feeling overwhelmed...')",
        height=150,
        key="post_input",
    )
    char_count_placeholder = st.empty()
    if char_limit_enabled:
        token_count = len(tokenizer.tokenize(user_input)) if user_input.strip() else 0
        char_count_placeholder.markdown(
            f"""
        <div class='tooltip'>
            <span id='char-count'>{len(user_input)}/{char_limit}</span>
            <span id='token-count'>Tokens: {token_count}</span>
            <span class='tooltiptext'>Limits input to {char_limit} characters, typical for social media posts. Adjust or disable in settings.</span>
        </div>
        """,
            unsafe_allow_html=True,
        )
    if st.button("Analyze"):
        if user_input.strip() == "":
            st.error("Please enter a post to analyze.")
        else:
            if char_limit_enabled and len(user_input) > char_limit:
                st.error(
                    f"Post exceeds {char_limit} characters. Please shorten it or adjust/disable the character limit."
                )
            else:
                with st.spinner("Analyzing..."):
                    sentiment, confidence = predict_sentiment(user_input, max_length)
                    if confidence >= confidence_threshold:
                        st.session_state.history.append(
                            {
                                "post": user_input,
                                "sentiment": sentiment,
                                "confidence": confidence,
                            }
                        )
                        with st.container():
                            st.markdown(
                                f"""
                            <div class='card'>
                                <h3>Sentiment Prediction</h3>
                                <p><b>Post:</b> {user_input}</p>
                                <p><b>Sentiment:</b> {sentiment}</p>
                                {f"<p><b>Confidence:</b> {confidence:.2%}</p>" if debug_mode else ""}
                                {f"<div class='progress-bar'><div class='progress-fill' style='width: {confidence * 100}%'></div></div>" if debug_mode else ""}
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )
                    else:
                        st.warning(
                            f"Prediction confidence ({confidence:.2%}) is below threshold ({confidence_threshold:.2%})."
                        )

with col2:
    with st.expander("Recent Predictions", expanded=True):
        if st.session_state.history:
            sentiment_filter = st.selectbox(
                "Filter by Sentiment", ["All"] + list(label_encoder.classes_)
            )
            for item in st.session_state.history[-5:][::-1]:
                if sentiment_filter == "All" or item["sentiment"] == sentiment_filter:
                    st.markdown(
                        f"""
                    <div class='history-item'>
                        <p><b>Post:</b> {item['post'][:50]}{'...' if len(item['post']) > 50 else ''}</p>
                        <p><b>Sentiment:</b> {item['sentiment']}</p>
                        {f"<p><b>Confidence:</b> {item['confidence']:.2%}</p>" if debug_mode else ""}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown("No predictions yet.")
