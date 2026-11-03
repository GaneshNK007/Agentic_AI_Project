# 🛡️ Agentic AI-Based Medical Assistant

An end-to-end Computer-Aided Diagnosis (CAD) and Clinical Decision Support System (CDSS) for **Primary Brain Tumor Classification** (Cranial MRI) and **Metastatic Tissue Screening** (Digital Histopathology Patches).

The system integrates deep convolutional neural networks (ResNet-50) with an **Agentic Orchestration Layer**, calibrated **Uncertainty Quantification (Normalized Shannon Entropy)**, dense vector **Retrieval-Augmented Generation (RAG)** via FAISS, and guardrailed clinical report synthesis powered by **Gemini 2.5 Flash**.

---

## 🌟 Key Features

1. **Dual-Modality Diagnostic Screening:**
   - **Brain MRI Scan Analysis:** Multi-class classification of primary intracranial neoplasms (*Glioma, Meningioma, Pituitary Tumor, Normal Brain*).
   - **Metastatic Tissue Screening:** Binary histopathology screening on sentinel lymph node patches (CAMELYON16 / PCam breast cancer benchmark).
2. **Agentic Decision Architecture:**
   - Dynamic input intake routing based on image characteristics.
   - Uncertainty-triggered specialist escalation: When classification ambiguity is high ($H \ge 0.65$), the agent intercepts the automated output, halts overconfident claims, and mandates specialist pathologist/radiologist review.
   - Adaptive RAG query reformulation (retrieving confirmatory IHC protocols like Cytokeratin AE1/AE3 or IDH1/GFAP on uncertain cases).
3. **Clinical Knowledge Grounding (RAG):**
   - Dense FAISS vector database (`all-MiniLM-L6-v2`) indexing guidelines from the **2021 WHO Classification of Tumors of the Central Nervous System (WHO CNS5)** and the **AJCC Cancer Staging Manual (8th Edition)**.
4. **Interactive CAD Portal:**
   - Real-time diagnostic dashboard with preprocessing adjustments (brightness, contrast, ImageNet normalization).
   - Conversational AI clinical assistant and searchable clinical knowledge library.

---

## 🏗️ Project Architecture

```
brain_tumor_agent/
├── backend/
│   ├── agent/
│   │   └── agent.py              # Main Agentic orchestrator & decision logic
│   ├── llm/
│   │   └── gemini.py             # Grounded report synthesis & validation
│   ├── models/
│   │   ├── model.py              # Brain MRI ResNet-50 inference & entropy
│   │   ├── metaCancer.py         # PCam Histopathology inference & entropy
│   │   └── statistical_eval.py   # Statistical significance & CI evaluation
│   ├── rag/
│   │   ├── retriever.py          # FAISS dense vector retriever
│   │   ├── eval_rag.py           # Quantitative RAG evaluation suite
│   │   └── eval_rag_expanded.py  # 60-case 3-way ablation benchmark
│   └── app.py                    # FastAPI REST API endpoints
├── data/
│   ├── medical_docs.txt          # Curated clinical literature corpus
│   └── samples/                  # Preloaded clinical demo sample scans
├── frontend/
│   └── app.py                    # Streamlit Interactive CAD Portal
├── requirements.txt              # Project dependencies
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/GaneshNK007/brain_tumor_agent.git
cd brain_tumor_agent

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Launch the Backend API
```bash
uvicorn backend.app:app --reload --port 8000
```
API Documentation will be accessible at `http://localhost:8000/docs`.

### 4. Launch the Interactive CAD Frontend
In a new terminal window:
```bash
streamlit run frontend/app.py
```
The web dashboard will open automatically in your browser at `http://localhost:8501`.

---

## ⚖️ Clinical Disclaimer
This software is an investigational Computer-Aided Decision Support System (CDSS) intended for research and educational purposes. It is not an independent diagnostic medical device. All outputs, confidence metrics, and synthesized reports must be evaluated and verified by certified medical practitioners.

