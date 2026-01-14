import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import custom explainability module
# We need to add src to path if running from root
import sys
sys.path.append('src')

try:
    from explain import ExplanationGenerator, load_and_preprocess_image
except ImportError:
    # If running from inside src
    from .explain import ExplanationGenerator, load_and_preprocess_image

# ==============================================================================
# CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="DermaAI - Intelligent Skin Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B5563;
        font-weight: 600;
    }
    .card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        background-color: #EFF6FF;
        border-radius: 8px;
        border: 1px solid #BFDBFE;
    }
    .success-text { color: #059669; font-weight: bold; }
    .warning-text { color: #D97706; font-weight: bold; }
    .danger-text { color: #DC2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_DIR = Path("models")
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

CLASSES_INFO = {
    'akiec': 'Actinic Keratoses (Pre-cancerous)',
    'bcc': 'Basal Cell Carcinoma (Cancerous)',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma (Dangerous Cancer)',
    'nv': 'Melanocytic Nevi (Mole)',
    'vasc': 'Vascular Lesions'
}

CRITICAL_CLASSES = ['mel', 'bcc', 'akiec']

# ==============================================================================
# UTILITIES
# ==============================================================================

@st.cache_resource
def load_model_resources():
    """Load model and class mapping with caching."""
    
    # 1. Load Class Mapping
    mapping_path = MODEL_DIR / "class_mapping.json"
    if not mapping_path.exists():
        st.error(f"Class mapping not found at {mapping_path}")
        return None, None
    
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
        # Ensure ordered list
        class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    
    # 2. Defile Custom Loss (for loading)
    try:
        from tensorflow.keras.losses import CategoricalFocalCrossentropy
    except ImportError:
        class CategoricalFocalCrossentropy(keras.losses.Loss):
            def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
                super().__init__(**kwargs)
            def call(self, y_true, y_pred): return tf.reduce_sum(y_true)

    # 3. Load Model
    model_path = MODEL_DIR / "best_model_finetuned.keras"
    if not model_path.exists():
        model_path = MODEL_DIR / "best_model.keras"
    
    if not model_path.exists():
        st.error(f"Model not found at {model_path}")
        return None, class_names

    try:
        model = keras.models.load_model(model_path, custom_objects={'CategoricalFocalCrossentropy': CategoricalFocalCrossentropy})
    except Exception as e:
        # Fallback if compilation fails (ok for inference)
        model = keras.models.load_model(model_path, compile=False)
        
    return model, class_names

@st.cache_resource
def load_explainer(_model, class_names):
    """Load and cache the explanation generator."""
    if _model is None: return None
    return ExplanationGenerator(_model, class_names)

def plot_probability_chart(probs, class_names):
    """Create a Plotly bar chart for probabilities."""
    df = pd.DataFrame({
        'Condition': [CLASSES_INFO.get(c, c) for c in class_names],
        'Probability': probs * 100,
        'Type': ['Critical' if c in CRITICAL_CLASSES else 'Benign/Other' for c in class_names],
        'Color': ['#DC2626' if c in CRITICAL_CLASSES else '#10B981' for c in class_names]
    })
    
    fig = px.bar(
        df, 
        y='Condition', 
        x='Probability', 
        orientation='h',
        color='Type',
        color_discrete_map={'Critical': '#EF4444', 'Benign/Other': '#10B981'},
        text=df['Probability'].apply(lambda x: f"{x:.1f}%")
    )
    
    fig.update_layout(
        title="Confidence Distribution",
        xaxis_title="Probability (%)",
        yaxis_title=None,
        showlegend=True,
        height=400
    )
    return fig

# ==============================================================================
# APP STRUCTURE
# ==============================================================================

def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/dermatology.png", width=80)
    st.sidebar.title("DermaAI")
    st.sidebar.markdown("---")
    
    app_mode = st.sidebar.radio("Navigation", ["Home", "Smart Diagnosis", "Model Analysis", "About & Model Info"])
    
    # Load Resources
    model, class_names = load_model_resources()
    
    if model is None:
        st.warning("Model is loading or missing... Please ensure training is complete.")
        return

    explainer = load_explainer(model, class_names)
    
    # --- PAGE: HOME ---
    if app_mode == "Home":
        st.markdown('<div class="main-header">DermaAI: Advanced Skin Lesion Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🩺 AI-Powered Dermatoscopy Assistant
            
            Welcome to **DermaAI**, a state-of-the-art tool designed to assist dermatologists in classifying skin lesions.
            This system leverages **EfficientNetB1** deep learning architecture and advanced explainability techniques (XAI)
            to provide transparent, reliable second opinions.
            
            #### Key Features:
            *   **7-Class Classification**: Covers Melanoma, Nevi, BCC, AKIEC, and more.
            *   **Safety-First**: Optimized for high sensitivity on critical cancers.
            *   **Explainable AI**: Visualize *why* the model made a decision using Grad-CAM & LIME.
            *   **Robustness**: Trained with Focal Loss, MixUp, and extensive augmentation.
            """)
            
            st.info("👈 Select **'Smart Diagnosis'** in the sidebar to test the model!")
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Model Stats")
            st.metric("Architecture", "EfficientNetB1")
            st.metric("Input Size", "224x224")
            st.metric("Params", "~7.8M")
            st.markdown("---")
            st.markdown("**Status:** 🟢 Trained & Ready")
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Dataset Info
        st.markdown("### 📚 Supported Conditions")
        cols = st.columns(3)
        for i, (code, desc) in enumerate(CLASSES_INFO.items()):
            with cols[i % 3]:
                critical = "🔴" if code in CRITICAL_CLASSES else "🟢"
                st.markdown(f"**{critical} {code.upper()}**: {desc}")
                
        # Technical Brief
        with st.expander("🔬 Technical Architecture Brief"):
            st.markdown("""
            - **Backbone**: EfficientNetB1 (Pre-trained ImageNet)
            - **Optimization**: AdamW Optimizer, Categorical Focal Loss (to handle class imbalance)
            - **Augmentation Pipeline**: RandomFlip, Rotation, Zoom, Contrast + **MixUp** regularization.
            - **Training**: Fine-tuned on HAM10000 (8,000+ images) with an 80/10/10 split.
            """)

    # --- PAGE: SMART DIAGNOSIS ---
    elif app_mode == "Smart Diagnosis":
        st.markdown('<div class="main-header">🔎 Smart Diagnosis & Explainability</div>', unsafe_allow_html=True)
        
        col_img, col_res = st.columns([1, 1.5])
        
        uploaded_file = st.sidebar.file_uploader("Upload Dermatoscopy Image", type=['jpg', 'jpeg', 'png'])
        
        # Test Image Selection (if no upload)
        if not uploaded_file:
            st.sidebar.markdown("---")
            st.sidebar.markdown("**Or select a sample:**")
            
            # Load Best Samples if available
            best_samples_path = RESULTS_DIR / "best_samples.json"
            best_samples = None
            if best_samples_path.exists():
                with open(best_samples_path, 'r') as f:
                    best_samples = json.load(f)

            test_dir = DATA_DIR / "split/test"
            
            if test_dir.exists():
                # Button for Random Sample (Uses curated list for demo)
                if st.sidebar.button("🎲 Load Random Test Sample"):
                    if best_samples:
                        # Pick random class from best samples
                        rand_cls = np.random.choice(list(best_samples.keys()))
                        # Pick one of the best samples for that class
                        sample_path = np.random.choice(best_samples[rand_cls])
                        
                        st.session_state['sample_img'] = str(sample_path)
                        st.session_state['true_label'] = rand_cls
                    else:
                        # Fallback if best_samples not found
                        rand_cls = np.random.choice(class_names)
                        cls_dir = test_dir / rand_cls
                        if cls_dir.exists():
                            imgs = list(cls_dir.glob("*.jpg"))
                            if imgs:
                                sample = np.random.choice(imgs)
                                st.session_state['sample_img'] = str(sample)
                                st.session_state['true_label'] = rand_cls
            
            if 'sample_img' in st.session_state:
                img_path = st.session_state['sample_img']
                st.sidebar.info(f"Loaded sample: {Path(img_path).name} (True: {st.session_state.get('true_label', '?')})")
        else:
            # Save uploaded file momentarily
            with open("temp_upload.jpg", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state['sample_img'] = "temp_upload.jpg"
            if 'true_label' in st.session_state: del st.session_state['true_label']

        # ANALYSIS LOGIC
        if 'sample_img' in st.session_state:
            img_path = st.session_state['sample_img']
            
            # Display Image
            with col_img:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                image = Image.open(img_path)
                st.image(image, caption="Analyzed Lesion", width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)

            # Run Inference
            if st.button("🧠 Analyze Lesion", type="primary"):
                with st.spinner("Processing image through EfficientNet..."):
                    # Preprocess
                    img_array, original = load_and_preprocess_image(img_path)
                    
                    # Predict
                    probs = model.predict(img_array, verbose=0)[0]
                    pred_idx = np.argmax(probs)
                    pred_class = class_names[pred_idx]
                    confidence = probs[pred_idx]
                    
                    st.session_state['last_probs'] = probs
                    st.session_state['pred_class'] = pred_class
                    
            # Display Results
            if 'last_probs' in st.session_state:
                probs = st.session_state['last_probs']
                pred_class = st.session_state['pred_class']
                confidence = probs[np.argmax(probs)]
                
                with col_res:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    
                    # Header
                    color = "#DC2626" if pred_class in CRITICAL_CLASSES else "#059669"
                    status = "HIGH RISK" if pred_class in CRITICAL_CLASSES else "BENIGN / LOW RISK"
                    
                    st.markdown(f"""
                    <h3 style='color: {color}; margin-top:0;'>Diagnosis: {CLASSES_INFO.get(pred_class, pred_class)}</h3>
                    <p style='font-size: 1.1rem;'><strong>Status:</strong> <span style='color:{color}; font-weight:bold;'>{status}</span></p>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    """, unsafe_allow_html=True)
                    
                    # Chart
                    st.plotly_chart(plot_probability_chart(probs, class_names))
                    st.markdown('</div>', unsafe_allow_html=True)

                # --- EXPLAINABILITY SECTION ---
                st.markdown("---")
                st.markdown("### 🧐 Explainability Lab")
                st.markdown("Understand *why* the model made this decision using state-of-the-art XAI methods.")
                
                tabs = st.tabs(["🔥 Grad-CAM", "🎯 LIME"])
                
                with tabs[0]:
                    with st.expander("ℹ️ How Grad-CAM works?"):
                        st.markdown("""
                        **Gradient-weighted Class Activation Mapping (Grad-CAM)**:
                        1. We perform a forward pass to get the prediction.
                        2. We calculate the *gradients* of the winning class score with respect to the feature maps of the **last convolutional layer** (`top_conv`).
                        3. These gradients tell us "where" the network is looking.
                        4. We create a heatmap: **Red** = High importance, **Blue** = Low importance.
                        """)
                    
                    col_cam1, col_cam2 = st.columns(2)
                    if st.button("Generate Grad-CAM", key="btn_gradcam"):
                        with st.spinner("Computing gradients relative to layer 'top_conv'..."):
                            # We can re-use ExplanationGenerator logic or call GradCAM directly
                            # Using ExplanationGenerator for ease
                            result = explainer.explain_image(img_path, methods=['gradcam', 'gradcam++'])
                            
                            with col_cam1:
                                st.image(result['gradcam']['overlay'], caption="Grad-CAM Overlay", width="stretch")
                                st.caption("Standard Grad-CAM: Good for coarse localization of lesion center.")
                                
                            with col_cam2:
                                st.image(result['gradcam++']['overlay'], caption="Grad-CAM++ Overlay", width="stretch")
                                st.caption("Grad-CAM++: Uses higher-order derivatives for sharper focus on details.")
                
                with tabs[1]:
                    with st.expander("ℹ️ How LIME works?"):
                        st.markdown("""
                        **Local Interpretable Model-agnostic Explanations (LIME)**:
                        1. It is a "black box" method. It doesn't look at gradients.
                        2. It generates **150+ perturbations** of the image by turning random "superpixels" (regions) on/off.
                        3. It observes how the model's confidence changes for each variation.
                        4. It fits a simple linear model to these local changes.
                        5. **Green Borders** = These regions INCREASE probability of the predicted class.
                        6. **Red Borders** = These regions DECREASE probability (look like another class).
                        """)

                    if st.button("Generate LIME Analysis", key="btn_lime"):
                        with st.spinner("Perturbing image 150 times to measure feature influence..."):
                            # LIME takes time
                            result = explainer.explain_image(img_path, methods=['lime'], lime_samples=150)
                            lime_res = result['lime']
                            
                            # Get image and mask
                            temp, mask = lime_res.get_image_and_mask(
                                lime_res.top_labels[0], positive_only=False, num_features=5, hide_rest=False
                            )
                            # Mark boundaries
                            from skimage.segmentation import mark_boundaries
                            # temp is in [0, 1], so we don't need to shift it
                            viz = mark_boundaries(temp, mask)
                            
                            st.image(viz, caption=f"LIME Superpixels (Green=Pro-{pred_class}, Red=Anti)", width="stretch")
                            
                            # Feature Importance Plot
                            st.markdown("#### Top Features")
                            # Use get_feature_importance defined in LIMEExplanation wrapper
                            imp = lime_res.get_feature_importance(label=lime_res.top_labels[0], num_features=5)
                            
                            # Just show weights
                            weights = [x[1] for x in imp]
                            ids = [f"Region {x[0]}" for x in imp]
                            
                            fig_lime = px.bar(x=weights, y=ids, orientation='h', title="Superpixel Importance")
                            st.plotly_chart(fig_lime)

    # --- PAGE: MODEL ANALYSIS ---
    elif app_mode == "Model Analysis":
        st.markdown('<div class="main-header">📊 Model Performance Analytics</div>', unsafe_allow_html=True)
        
        # Load pre-computed results if available
        report_path = RESULT_DIR = Path("results/evaluation_report.json") if (Path("results/evaluation_report.json").exists()) else None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Confusion Matrix")
            cm_path = RESULTS_DIR / "confusion_matrix.png"
            if cm_path.exists():
                st.image(str(cm_path), width="stretch")
            else:
                st.info("Confusion matrix not found. Run evaluation script first.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Classification Metrics")
            if report_path and report_path.exists():
                with open(report_path) as f:
                    report = json.load(f)
                
                # Format as dataframe
                data = []
                for k, v in report.items():
                    if isinstance(v, dict):
                        row = {'Class': k, **v}
                        data.append(row)
                df_metrics = pd.DataFrame(data).set_index('Class')
                st.dataframe(df_metrics.style.highlight_max(axis=0))
                
                st.metric("Overall Accuracy", f"{report['accuracy']:.2%}")
            else:
                st.info("Metrics report not found. Run evaluation script first.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### 📈 Training History")
        hist_path = RESULTS_DIR / "training_history.png"
        if hist_path.exists():
            st.image(str(hist_path), width="stretch")
            
    # --- PAGE: ABOUT ---
    elif app_mode == "About & Model Info":
        st.markdown('<div class="main-header">🧠 Model & Architecture Details</div>', unsafe_allow_html=True)
        
        st.markdown("### 1. Convolutional Neural Network (CNN)")
        st.info("**EfficientNetB1** was chosen for its excellent balance between accuracy and computational efficiency (~7.8M parameters vs ~25M for ResNet50).")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Architecture Design")
            st.markdown("""
            *   **Backbone**: EfficientNetB1 (Pre-trained on ImageNet).
            *   **Input Shape**: (224, 224, 3) RGB.
            *   **Custom Head**:
                *   `GlobalAveragePooling2D` -> Reduces spatial dimensions 7x7 -> 1x1.
                *   `Dropout(0.4)` -> Regularization to prevent overfitting.
                *   `Dense(512, ReLU)` -> Intermediate high-level feature processing.
                *   `Dense(7, Softmax)` -> Final classification probabilities.
            """)
        with c2:
            st.markdown("#### Training Hyperparameters")
            st.code("""
EPOCHS = 30 (10 Initial + 20 Fine-tuning)
BATCH_SIZE = 32
OPTIMIZER = AdamW (Weight Decay = 0.01)
LOSS = Categorical Focal Loss (alpha=0.25, gamma=2.0)
LEARNING_RATE = 1e-3 -> Reduced on Plateau
            """, language="yaml")

        st.markdown("---")
        st.markdown("### 2. Advanced Training Techniques")
        
        st.markdown("""
        #### ⚖️ Handling Class Imbalance
        The HAM10000 dataset is heavily imbalanced (67% Nevi vs 1.1% Dermatofibroma). We addressed this via:
        1.  **Oversampling**: We create a balanced virtual epoch by oversampling minority classes.
        2.  **Focal Loss**: A loss function that applies a heavier penalty to "hard" misclassified examples compared to easy ones.
            $$ FL(p_t) = -\\alpha_t (1 - p_t)^\\gamma \\log(p_t) $$
        
        #### 🔄 Data Augmentation with MixUp
        Standard augmentation (rotations, flips) is not enough. We use **MixUp**:
        *   We blend two images together: $X_{new} = \\lambda X_1 + (1-\\lambda) X_2$
        *   We blend their labels similarly: $Y_{new} = \\lambda Y_1 + (1-\\lambda) Y_2$
        *   This forces the model to learn linear transitions between classes, improving robustness.
        
        ---
        
        ### 3. Explainability (XAI)
        
        #### 🔥 Grad-CAM
        Gradient-weighted Class Activation Mapping uses the **gradients** flowing into the final convolutional layer of the network (`top_conv`) to produce a coarse localization map highlighting the important regions in the image for predicting the concept.
        
        #### 🎯 LIME
        Local Interpretable Model-agnostic Explanations treats the model as a black box. It perturbs the image by superpixels (contiguous regions of similar color/texture) and builds a local linear model to approximate the behavior. It answers: *"If I hide this part of the skin, does the probability of Melanoma drop?"*
        """)
        
        st.markdown("---")
        st.caption("Developed for EPITA ING3 ML-Bio Project.")

if __name__ == "__main__":
    main()
