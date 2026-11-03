import streamlit as st
import requests
import io
import os
import base64
from PIL import Image, ImageEnhance, ImageDraw

# ==========================================
# 🎨 PREMIUM CSS THEME INJECTION
# ==========================================
st.set_page_config(page_title="CAD - Medical AI Agent", layout="wide", page_icon="🛡️")

custom_css = """
<style>
    /* Main App Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f172a;
        color: #f1f5f9;
    }
    
    /* Header styling */
    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Custom Card container */
    .cad-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Metric Card Styling */
    .metric-container {
        display: flex;
        gap: 15px;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .metric-box {
        flex: 1;
        background: #1e293b;
        border: 1px solid #334155;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
        letter-spacing: 0.05em;
    }
    .metric-val {
        font-size: 22px;
        font-weight: 700;
        color: #38bdf8;
    }
    
    /* Risk Levels color badges */
    .risk-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 14px;
        text-align: center;
        width: 100%;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .risk-badge-low {
        background-color: rgba(16, 185, 129, 0.12);
        border: 1px solid #10b981;
        color: #34d399;
    }
    .risk-badge-medium {
        background-color: rgba(245, 158, 11, 0.12);
        border: 1px solid #f59e0b;
        color: #fbbf24;
    }
    .risk-badge-high {
        background-color: rgba(239, 68, 68, 0.12);
        border: 1px solid #ef4444;
        color: #f87171;
    }
    
    /* Modality badge */
    .modality-pill {
        display: inline-block;
        background-color: #334155;
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    /* Hide Streamlit components */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 🖼️ DYNAMIC MOCK SAMPLE GENERATOR
# ==========================================
def create_sample_images():
    os.makedirs("data/samples", exist_ok=True)
    
    # 1. Normal MRI
    normal_mri_path = "data/samples/normal_mri.png"
    if not os.path.exists(normal_mri_path):
        img = Image.new("RGB", (224, 224), color=(10, 10, 10))
        draw = ImageDraw.Draw(img)
        draw.ellipse([30, 30, 194, 194], outline=(100, 100, 100), width=3)
        draw.ellipse([80, 80, 110, 140], fill=(25, 25, 25))
        draw.ellipse([114, 80, 144, 140], fill=(25, 25, 25))
        draw.ellipse([97, 100, 127, 150], fill=(15, 15, 15))
        img.save(normal_mri_path)
        
    # 2. Glioma MRI
    glioma_mri_path = "data/samples/glioma_mri.png"
    if not os.path.exists(glioma_mri_path):
        img = Image.new("RGB", (224, 224), color=(10, 10, 10))
        draw = ImageDraw.Draw(img)
        draw.ellipse([30, 30, 194, 194], outline=(100, 100, 100), width=3)
        draw.ellipse([80, 80, 110, 140], fill=(25, 25, 25))
        draw.ellipse([114, 80, 144, 140], fill=(25, 25, 25))
        # Glioma tumor (bright spot)
        draw.ellipse([60, 60, 110, 110], fill=(200, 200, 200), outline=(255, 255, 255), width=2)
        draw.ellipse([50, 50, 120, 120], outline=(100, 100, 100), width=1)
        img.save(glioma_mri_path)
        
    # 3. Normal Tissue (Histopathology)
    normal_tissue_path = "data/samples/normal_tissue.png"
    if not os.path.exists(normal_tissue_path):
        img = Image.new("RGB", (96, 96), color=(240, 200, 220))
        draw = ImageDraw.Draw(img)
        for i in range(15):
            x = (i * 23 + 17) % 80 + 8
            y = (i * 37 + 11) % 80 + 8
            draw.ellipse([x, y, x+6, y+6], fill=(120, 50, 120))
        img.save(normal_tissue_path)
        
    # 4. Cancer Tissue (Histopathology with tumor cluster)
    cancer_tissue_path = "data/samples/cancer_tissue.png"
    if not os.path.exists(cancer_tissue_path):
        img = Image.new("RGB", (96, 96), color=(230, 185, 210))
        draw = ImageDraw.Draw(img)
        # Dense metastatic tumor cellular cluster in center 32x32
        for i in range(50):
            x = (i * 11 + 7) % 36 + 30
            y = (i * 17 + 5) % 36 + 30
            size = (i % 5) + 6
            draw.ellipse([x, y, x+size, y+size], fill=(70, 10, 90), outline=(40, 5, 60))
        # Peripheral lymphoid cells
        for i in range(25):
            x = (i * 19 + 3) % 90 + 3
            y = (i * 29 + 13) % 90 + 3
            draw.ellipse([x, y, x+4, y+4], fill=(130, 60, 130))
        img.save(cancer_tissue_path)

create_sample_images()

# ==========================================
# 🛡️ MAIN LAYOUT & NAVIGATION
# ==========================================
st.title("🛡️ Interactive Computer-Aided Diagnosis (CAD) Portal")
st.markdown("#### Multi-Modal Clinical Diagnostic Assistant powered by Deep Learning & Retrieval-Augmented Generation (RAG)")

tab_dashboard, tab_chat, tab_knowledge = st.tabs([
    "🖥️ Diagnostic Dashboard",
    "💬 AI Clinical Assistant",
    "📚 Clinical Knowledge Library"
])


# ==========================================
# 🖥️ TAB: DIAGNOSTIC DASHBOARD
# ==========================================
with tab_dashboard:
    # Sub-tabs for the two modalities
    subtab_brain, subtab_meta = st.tabs([
        "🧠 Brain Tumor Analysis (MRI)",
        "🔬 Metastatic Tissue Screening (PCam Lymph Node Histopathology)"
    ])

    # -------------------------------------------------------------
    # SUBTAB 1: 🧠 BRAIN TUMOR ANALYSIS (MRI)
    # -------------------------------------------------------------
    with subtab_brain:
        col_brain_in, col_brain_out = st.columns([1, 1.2])

        with col_brain_in:
            st.markdown('<div class="cad-card">', unsafe_allow_html=True)
            st.markdown('<span class="modality-pill">Modality: Brain MRI &bull; Model: ResNet-50</span>', unsafe_allow_html=True)
            st.subheader("📷 Brain MRI Scan Intake")

            brain_source = st.radio("Intake Source", ["Upload File", "Choose from Samples"], key="brain_source", horizontal=True)

            brain_pil = None
            brain_filename = ""

            if brain_source == "Upload File":
                brain_upload = st.file_uploader("Upload Brain MRI Scan", type=["jpg", "jpeg", "png"], key="brain_uploader")
                if brain_upload:
                    brain_pil = Image.open(brain_upload).convert("RGB")
                    brain_filename = brain_upload.name
            else:
                brain_sample_opt = st.selectbox(
                    "Select Sample Brain Scan",
                    ["Healthy Brain MRI", "Glioma Brain MRI"],
                    key="brain_sample_select"
                )
                sample_map = {
                    "Healthy Brain MRI": "data/samples/normal_mri.png",
                    "Glioma Brain MRI": "data/samples/glioma_mri.png"
                }
                sample_path = sample_map[brain_sample_opt]
                brain_pil = Image.open(sample_path)
                brain_filename = os.path.basename(sample_path)

            st.markdown("---")
            st.subheader("⚙️ Preprocessing & Adjustments")
            brain_bright = st.slider("Brightness Scale", 0.5, 2.0, 1.0, step=0.1, key="brain_bright")
            brain_contrast = st.slider("Contrast Scale", 0.5, 2.0, 1.0, step=0.1, key="brain_contrast")
            brain_norm = st.checkbox(
                "⚙️ Apply standard ImageNet Normalization",
                value=True,
                key="brain_norm",
                help="Toggles between normalized [-2.1, 2.6] inputs and unnormalized [0.0, 1.0] inputs."
            )

            brain_enhanced = None
            if brain_pil:
                enhancer = ImageEnhance.Brightness(brain_pil)
                temp = enhancer.enhance(brain_bright)
                enhancer_c = ImageEnhance.Contrast(temp)
                brain_enhanced = enhancer_c.enhance(brain_contrast)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(brain_pil, caption="Original Intake Scan", use_container_width=True)
                with c2:
                    st.image(brain_enhanced, caption="Preprocessed Scan (sent to Model)", use_container_width=True)
            else:
                st.info("💡 Please upload a brain MRI scan or choose a sample scan to view preprocessing controls.")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_brain_out:
            st.markdown('<div class="cad-card">', unsafe_allow_html=True)
            st.subheader("📊 Diagnostic Diagnosis Output")

            if brain_enhanced:
                brain_run_btn = st.button("🚀 Analyze Brain Scan", type="primary", use_container_width=True, key="brain_run_btn")

                if brain_run_btn:
                    with st.spinner("⚡ Running Deep Learning Classifier & Retrieval-Augmented Generation..."):
                        try:
                            img_byte_arr = io.BytesIO()
                            brain_enhanced.save(img_byte_arr, format='PNG')
                            img_byte_arr.seek(0)

                            res = requests.post(
                                "http://127.0.0.1:8000/agent",
                                files={"file": (brain_filename, img_byte_arr, "image/png")},
                                params={"image_type": "brain", "normalize": brain_norm}
                            )

                            if res.status_code == 200:
                                result = res.json()
                                if "error" in result:
                                    st.error(f"❌ Model Error: {result['error']}")
                                else:
                                    pred = result.get("prediction", "Unknown")
                                    conf = result.get("confidence", 0.0)
                                    uncertainty = result.get("uncertainty", 0.0)
                                    risk = result.get("risk", "Low")
                                    explanation = result.get("explanation", "No explanation available.")
                                    reference = result.get("clinical_reference", "No reference retrieved.")
                                    gradcam_base64 = result.get("gradcam", "")

                                    # Save to session state
                                    st.session_state.last_diagnosis = {
                                        "prediction": pred,
                                        "confidence": conf,
                                        "uncertainty": uncertainty,
                                        "risk": risk,
                                        "explanation": explanation,
                                        "reference": reference,
                                        "scan_category": "Brain MRI",
                                        "modality": "Brain MRI Scan",
                                        "filename": brain_filename
                                    }

                                    # Clinical Report Card
                                    st.markdown(f"""
                                    <div style="border: 2px solid #334155; border-radius: 12px; padding: 24px; background-color: #1e293b; margin-top: 15px; margin-bottom: 15px;">
                                        <div style="text-align: center; border-bottom: 2px solid #38bdf8; padding-bottom: 15px; margin-bottom: 20px;">
                                            <h2 style="margin: 0; color: #38bdf8; font-size: 20px; letter-spacing: 0.05em;">🛡️ COMPUTER-AIDED DIAGNOSIS (CAD) CLINICAL REPORT</h2>
                                            <small style="color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; font-size: 10px;">Facility: Advanced Clinical AI Lab &bull; ID: CAD-MRI-2026-0815</small>
                                        </div>
                                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 13px; color: #cbd5e1; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 15px;">
                                            <div><strong>📁 Modality:</strong> Brain MRI Scan</div>
                                            <div><strong>📄 Filename:</strong> {brain_filename}</div>
                                            <div><strong>🔆 Brightness Scale:</strong> {brain_bright}x</div>
                                            <div><strong>🌗 Contrast Scale:</strong> {brain_contrast}x</div>
                                        </div>
                                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                                            <div style="background-color: #0f172a; padding: 12px; border-radius: 6px; text-align: center; border: 1px solid #334155;">
                                                <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">Classification</div>
                                                <div style="font-size: 18px; font-weight: 700; color: #38bdf8;">{pred}</div>
                                            </div>
                                            <div style="background-color: #0f172a; padding: 12px; border-radius: 6px; text-align: center; border: 1px solid #334155;">
                                                <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">Model Confidence</div>
                                                <div style="font-size: 18px; font-weight: 700; color: #34d399;">{conf*100:.2f}%</div>
                                            </div>
                                            <div style="background-color: #0f172a; padding: 12px; border-radius: 6px; text-align: center; border: 1px solid #334155; grid-column: span 2;">
                                                <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">AI Uncertainty Rating (Std Dev)</div>
                                                <div style="font-size: 18px; font-weight: 700; color: {'#f87171' if uncertainty > 0.15 else '#38bdf8'};">{uncertainty*100:.2f}%</div>
                                                <small style="color: #64748b; font-size: 10px;">{'(High uncertainty: prediction may be unstable)' if uncertainty > 0.15 else '(Low uncertainty: prediction is stable)'}</small>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    st.progress(conf)

                                    # Risk Badge
                                    risk_lower = risk.lower()
                                    if "high" in risk_lower or "clinical" in risk_lower or "tumor detected" in risk_lower:
                                        risk_class = "risk-badge-high"
                                    elif "medium" in risk_lower or "uncertain" in risk_lower:
                                        risk_class = "risk-badge-medium"
                                    else:
                                        risk_class = "risk-badge-low"
                                    st.markdown(f'<div class="risk-badge {risk_class}">⚠️ Risk Category: {risk}</div>', unsafe_allow_html=True)

                                    if gradcam_base64:
                                        st.markdown("---")
                                        st.markdown("### 🔬 Explainable AI: Grad-CAM Saliency Map")
                                        st.image(base64.b64decode(gradcam_base64), caption="Saliency Heatmap", use_container_width=True)

                                    st.markdown("---")
                                    st.markdown("### 📖 Generative Diagnostic Insight (Gemini)")
                                    st.markdown(explanation)

                                    st.markdown("### 📚 Clinician Reference Literature (RAG match)")
                                    st.info(reference)

                                    report_md = f"""# 🛡️ COMPUTER-AIDED DIAGNOSIS (CAD) CLINICAL REPORT
**Facility:** Advanced Clinical AI Lab
**Report ID:** CAD-MRI-2026-0815
**Modality:** Brain MRI Scan
**Intake Filename:** {brain_filename}
**Preprocessing Adjustments:** Brightness: {brain_bright}x | Contrast: {brain_contrast}x
---
## 📊 DIAGNOSTIC SCAN ANALYSIS
- **Model Classification:** **{pred}**
- **Classifier Confidence:** **{conf*100:.2f}%**
- **Model Uncertainty (Std Dev):** **{uncertainty*100:.2f}%**
- **Clinical Risk Category:** **{risk}**

---
## 📚 CLINICIAN REFERENCE LITERATURE (RAG match)
"{reference}"

---
## 🧠 DETAILED MEDICAL INTERPRETATION (AI Agent)
{explanation}

---
*Disclaimer: This is a computer-aided diagnostic report generated by AI. It is intended for clinical reference and decision support only, and must be reviewed by a certified medical specialist.*
"""
                                    st.download_button(
                                        label="📥 Download Clinical Report (Markdown)",
                                        data=report_md,
                                        file_name=f"Clinical_Report_Brain_MRI_{pred.replace(' ', '_')}.md",
                                        mime="text/markdown",
                                        use_container_width=True,
                                        key="brain_download_btn"
                                    )
                            else:
                                st.error(f"❌ Connection to backend failed with status code: {res.status_code}")
                        except Exception as e:
                            st.error(f"❌ Connection error: {str(e)}")
            else:
                st.info("🩺 Pending intake scan selection. Choose or upload a scan in the left panel to begin diagnostic analysis.")

            st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # SUBTAB 2: 🔬 METASTATIC TISSUE SCREENING ON HISTOPATHOLOGY PATCHES (PCAM)
    # -------------------------------------------------------------
    with subtab_meta:
        col_meta_in, col_meta_out = st.columns([1, 1.2])

        with col_meta_in:
            st.markdown('<div class="cad-card">', unsafe_allow_html=True)
            st.markdown('<span class="modality-pill">Modality: Sentinel Lymph Node Histopathology (96×96) &bull; Model: PCam ResNet-50</span>', unsafe_allow_html=True)
            st.caption("ℹ️ *Evaluates metastatic breast cancer infiltration in sentinel lymph nodes (CAMELYON16 benchmark). Completely decoupled from primary brain MRI.*")
            st.subheader("📷 Metastatic Tissue Patch Intake")

            meta_source = st.radio("Intake Source", ["Upload File", "Choose from Samples"], key="meta_source", horizontal=True)

            meta_pil = None
            meta_filename = ""

            if meta_source == "Upload File":
                meta_upload = st.file_uploader(
                    "Upload Histopathology Image Patch (96x96 / RGB)",
                    type=["jpg", "jpeg", "png"],
                    key="meta_uploader"
                )
                if meta_upload:
                    meta_pil = Image.open(meta_upload).convert("RGB")
                    meta_filename = meta_upload.name
            else:
                meta_sample_opt = st.selectbox(
                    "Select Sample Histopathology Scan",
                    ["Normal Lymph Node Tissue", "Metastatic Cancer Tissue"],
                    key="meta_sample_select"
                )
                sample_map_meta = {
                    "Normal Lymph Node Tissue": "data/samples/normal_tissue.png",
                    "Metastatic Cancer Tissue": "data/samples/cancer_tissue.png"
                }
                sample_path = sample_map_meta[meta_sample_opt]
                meta_pil = Image.open(sample_path)
                meta_filename = os.path.basename(sample_path)

            st.markdown("---")
            st.subheader("⚙️ Preprocessing & Adjustments")
            meta_bright = st.slider("Brightness Scale", 0.5, 2.0, 1.0, step=0.1, key="meta_bright")
            meta_contrast = st.slider("Contrast Scale", 0.5, 2.0, 1.0, step=0.1, key="meta_contrast")
            meta_norm = st.checkbox(
                "⚙️ Apply standard ImageNet Normalization",
                value=True,
                key="meta_norm",
                help="Applies standard ImageNet mean/std normalization for the PCam ResNet-50 backbone."
            )

            meta_enhanced = None
            if meta_pil:
                enhancer = ImageEnhance.Brightness(meta_pil)
                temp = enhancer.enhance(meta_bright)
                enhancer_c = ImageEnhance.Contrast(temp)
                meta_enhanced = enhancer_c.enhance(meta_contrast)

                c1, c2 = st.columns(2)
                with c1:
                    st.image(meta_pil, caption="Original Tissue Patch", use_container_width=True)
                with c2:
                    st.image(meta_enhanced, caption="Preprocessed Patch (sent to PCam Model)", use_container_width=True)
            else:
                st.info("💡 Please upload a histopathology image patch or choose a sample to view preprocessing controls.")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_meta_out:
            st.markdown('<div class="cad-card">', unsafe_allow_html=True)
            st.subheader("📊 Metastatic Diagnosis Output")

            if meta_enhanced:
                meta_run_btn = st.button("🚀 Analyze Metastatic Scan", type="primary", use_container_width=True, key="meta_run_btn")

                if meta_run_btn:
                    with st.spinner("⚡ Running PCam ResNet50 Classifier & Generative Clinical Reporting..."):
                        try:
                            img_byte_arr = io.BytesIO()
                            meta_enhanced.save(img_byte_arr, format='PNG')
                            img_byte_arr.seek(0)

                            res = requests.post(
                                "http://127.0.0.1:8000/agent",
                                files={"file": (meta_filename, img_byte_arr, "image/png")},
                                params={"image_type": "cancer", "normalize": meta_norm}
                            )

                            if res.status_code == 200:
                                result = res.json()
                                if "error" in result:
                                    st.error(f"❌ Model Error: {result['error']}")
                                else:
                                    pred = result.get("prediction", "Unknown")
                                    conf = result.get("confidence", 0.0)
                                    uncertainty = result.get("uncertainty", 0.0)
                                    risk = result.get("risk", "Low")
                                    explanation = result.get("explanation", "No explanation available.")
                                    reference = result.get("clinical_reference", "No reference retrieved.")
                                    all_probs = result.get("all_probabilities", {})

                                    # Save to session state
                                    st.session_state.last_diagnosis = {
                                        "prediction": pred,
                                        "confidence": conf,
                                        "uncertainty": uncertainty,
                                        "risk": risk,
                                        "explanation": explanation,
                                        "reference": reference,
                                        "scan_category": "Metastatic Tumor (PCam)",
                                        "modality": "Metastatic Tumor Histopathology (PCam)",
                                        "filename": meta_filename
                                    }

                                    # Clinical Report Card
                                    st.markdown(f"""
                                    <div style="border: 2px solid #334155; border-radius: 12px; padding: 24px; background-color: #1e293b; margin-top: 15px; margin-bottom: 15px;">
                                        <div style="text-align: center; border-bottom: 2px solid #38bdf8; padding-bottom: 15px; margin-bottom: 20px;">
                                            <h2 style="margin: 0; color: #38bdf8; font-size: 20px; letter-spacing: 0.05em;">🔬 COMPUTER-AIDED DIAGNOSIS: METASTATIC PATHOLOGY REPORT</h2>
                                            <small style="color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; font-size: 10px;">Facility: Advanced Clinical AI Lab &bull; Model: pcam.keras (PatchCamelyon ResNet50)</small>
                                        </div>
                                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; font-size: 13px; color: #cbd5e1; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 15px;">
                                            <div><strong>📁 Modality:</strong> Histopathology (Lymph Node)</div>
                                            <div><strong>📄 Filename:</strong> {meta_filename}</div>
                                            <div><strong>🔆 Brightness Scale:</strong> {meta_bright}x</div>
                                            <div><strong>🌗 Contrast Scale:</strong> {meta_contrast}x</div>
                                        </div>
                                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;">
                                            <div style="background-color: #0f172a; padding: 12px; border-radius: 6px; text-align: center; border: 1px solid #334155;">
                                                <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">Classification</div>
                                                <div style="font-size: 18px; font-weight: 700; color: {'#f87171' if 'cancer' in pred.lower() or 'metastatic' in pred.lower() else '#38bdf8'};">{pred}</div>
                                            </div>
                                            <div style="background-color: #0f172a; padding: 12px; border-radius: 6px; text-align: center; border: 1px solid #334155;">
                                                <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">Model Confidence</div>
                                                <div style="font-size: 18px; font-weight: 700; color: #34d399;">{conf*100:.2f}%</div>
                                            </div>
                                            <div style="background-color: #0f172a; padding: 12px; border-radius: 6px; text-align: center; border: 1px solid #334155; grid-column: span 2;">
                                                <div style="font-size: 11px; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px;">AI Uncertainty Rating (Std Dev)</div>
                                                <div style="font-size: 18px; font-weight: 700; color: {'#f87171' if uncertainty > 0.15 else '#38bdf8'};">{uncertainty*100:.2f}%</div>
                                                <small style="color: #64748b; font-size: 10px;">{'(High uncertainty: cellular morphology near decision threshold)' if uncertainty > 0.15 else '(Low uncertainty: clear morphological indicators)'}</small>
                                            </div>
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    st.progress(conf)

                                    # Risk Badge
                                    risk_lower = risk.lower()
                                    if "high" in risk_lower or "metastatic" in risk_lower:
                                        risk_class = "risk-badge-high"
                                    elif "medium" in risk_lower or "suspicious" in risk_lower:
                                        risk_class = "risk-badge-medium"
                                    else:
                                        risk_class = "risk-badge-low"
                                    st.markdown(f'<div class="risk-badge {risk_class}">⚠️ Risk Category: {risk}</div>', unsafe_allow_html=True)

                                    # Show class breakdown if available
                                    if all_probs:
                                        c_norm, c_meta = st.columns(2)
                                        with c_norm:
                                            st.metric("Normal Tissue Probability", f"{all_probs.get('Normal Tissue', 0.0)*100:.1f}%")
                                        with c_meta:
                                            st.metric("Metastatic Cancer Probability", f"{all_probs.get('Metastatic Cancer', 0.0)*100:.1f}%")

                                    st.markdown("---")
                                    st.markdown("### 📖 Generative Diagnostic Insight (Gemini)")
                                    st.markdown(explanation)

                                    st.markdown("### 📚 Clinician Reference Literature (RAG match)")
                                    st.info(reference)

                                    report_md = f"""# 🔬 COMPUTER-AIDED DIAGNOSIS: METASTATIC PATHOLOGY REPORT
**Facility:** Advanced Clinical AI Lab
**Model:** pcam.keras (ResNet-50 PatchCamelyon)
**Modality:** Histopathology Scan (Lymph Node Tissue)
**Intake Filename:** {meta_filename}
**Preprocessing Adjustments:** Brightness: {meta_bright}x | Contrast: {meta_contrast}x
---
## 📊 DIAGNOSTIC TISSUE ANALYSIS
- **Model Classification:** **{pred}**
- **Classifier Confidence:** **{conf*100:.2f}%**
- **Model Uncertainty (Std Dev):** **{uncertainty*100:.2f}%**
- **Clinical Risk Category:** **{risk}**

---
## 📚 CLINICIAN REFERENCE LITERATURE (RAG match)
"{reference}"

---
## 🧠 DETAILED MEDICAL INTERPRETATION (AI Agent)
{explanation}

---
*Disclaimer: This is a computer-aided diagnostic report generated by AI. It is intended for clinical reference and decision support only, and must be reviewed by a certified pathologist or oncologist.*
"""
                                    st.download_button(
                                        label="📥 Download Clinical Report (Markdown)",
                                        data=report_md,
                                        file_name=f"Clinical_Report_Metastatic_{pred.replace(' ', '_')}.md",
                                        mime="text/markdown",
                                        use_container_width=True,
                                        key="meta_download_btn"
                                    )
                            else:
                                st.error(f"❌ Connection to backend failed with status code: {res.status_code}")
                        except Exception as e:
                            st.error(f"❌ Connection error: {str(e)}")
            else:
                st.info("🩺 Pending intake patch selection. Choose or upload a histopathology image patch in the left panel to begin diagnostic analysis.")

            st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 💬 TAB: AI CLINICAL ASSISTANT
# ==========================================
with tab_chat:
    st.markdown('<div class="cad-card">', unsafe_allow_html=True)
    st.subheader("💬 AI Conversational Clinical Assistant")
    
    # Context Notice & Clear Button
    if "last_diagnosis" in st.session_state and st.session_state.last_diagnosis:
        ld = st.session_state.last_diagnosis
        st.info(f"🩺 **Diagnosis Context Active:** Chatbot is aware of the latest report context: **{ld['prediction']} ({ld['scan_category']})**.")
        grant_permission = st.checkbox("🔓 Grant AI Assistant permission to read this diagnostic report", value=True)
        if st.button("🧹 Clear Report Context"):
            st.session_state.last_diagnosis = None
            st.rerun()
    else:
        st.markdown("Ask general medical questions or check queries using the LangChain RAG-enabled chatbot.")
        grant_permission = False
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    chat_input = st.chat_input("Ask a clinical query (e.g. 'What is metastatic cancer?' or 'Explain the findings')")
    
    if chat_input:
        st.session_state.messages.append({"role": "user", "content": chat_input})
        with st.chat_message("user"):
            st.markdown(chat_input)
            
        # Prep prefix context if active and permitted
        context_prefix = ""
        if "last_diagnosis" in st.session_state and st.session_state.last_diagnosis and grant_permission:
            ld = st.session_state.last_diagnosis
            context_prefix = (
                f"Active Patient Diagnostic Report Context:\n"
                f"- Scan Modality: {ld['modality']}\n"
                f"- Classification Finding: {ld['prediction']}\n"
                f"- Confidence Rating: {ld['confidence']*100:.2f}%\n"
                f"- Risk Assessment: {ld['risk']}\n"
                f"- Generative Medical Report: {ld['explanation']}\n"
                f"- Clinical Reference matched: {ld['reference']}\n\n"
                f"User Question to address: "
            )
            
        full_query = context_prefix + chat_input
            
        with st.spinner("🤖 Consulting Clinical AI Agent..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/agent",
                    params={"query": full_query}
                )
                if res.status_code == 200:
                    result = res.json()
                    bot_response = result.get("answer", "⚠️ No answer received.")
                else:
                    bot_response = f"❌ Backend Error: Status Code {res.status_code}"
            except Exception as e:
                bot_response = f"❌ Error communicating with backend: {str(e)}"
                
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)
            
    st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# 📚 TAB: CLINICAL KNOWLEDGE LIBRARY
# ==========================================
with tab_knowledge:
    st.markdown('<div class="cad-card">', unsafe_allow_html=True)
    st.subheader("📚 Clinical Knowledge Library")
    st.markdown("Reference literature indexed in the FAISS Vector Database for Retrieval-Augmented Generation (RAG).")
    
    try:
        with open("data/medical_docs.txt", "r", encoding="utf-8") as f:
            docs = f.readlines()
        docs = [d.strip() for d in docs if d.strip()]
        
        for i, doc in enumerate(docs):
            if "brain" in doc.lower() or "glioma" in doc.lower() or "meningioma" in doc.lower() or "pituitary" in doc.lower():
                tag = "🧠 Brain MRI Literature"
            else:
                tag = "🔬 Metastatic Pathology Literature"
            st.markdown(f"**Record #{i+1} &bull; `{tag}`:**")
            st.info(doc)
    except Exception as e:
        st.error(f"❌ Failed to load medical docs: {str(e)}")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Debug checkbox
st.markdown("---")
if st.checkbox("🧪 Show Debug Panel"):
    st.write("Session State:")
    st.json(st.session_state)