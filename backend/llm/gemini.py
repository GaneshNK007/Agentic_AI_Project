import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)


def generate_medical_response(prediction, confidence, risk, user_query=None, context=None, modality=None):
    """
    Generates a structured medical report or answers clinical questions using Gemini 2.5 Flash.
    Supports grounding with clinical reference literature (RAG) and modality awareness.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")

    context_str = f"\nClinical Literature Reference:\n{context}\n" if context else ""
    modality_str = f"Diagnostic Modality: {modality}\n" if modality else ""

    if user_query:
        prompt = f"""
You are an expert Clinical AI Medical Assistant.
{context_str}
Answer the following clinical or patient query clearly, accurately, and compassionately:
{user_query}

Keep it professional, patient-friendly, and medically accurate. Ground your response in the provided clinical reference information if applicable.
"""
    else:
        prompt = f"""
You are a specialized Clinical AI Diagnostic Assistant.
{context_str}
{modality_str}Diagnostic Scan Analysis:
- Prediction / Finding: {prediction}
- AI Model Confidence: {confidence}
- Clinical Risk Assessment: {risk}

Please generate a comprehensive, structured clinical report with the following sections:
### 1. Diagnostic Summary & Biological Definition
Explain clearly what this classification ({prediction}) signifies, the nature of the tissue/cells examined, and whether malignancy or abnormal proliferation is present.

### 2. Clinical Implications & Pathology Insights
Discuss how this finding relates to disease progression, staging (e.g., lymph node metastasis or tumor grade), or benign confirmation.

### 3. Key Symptoms & Diagnostic Indicators
Identify common clinical symptoms or systemic signs associated with this condition that patients or clinicians should monitor.

### 4. Recommended Clinical Action Plan
Provide actionable next steps (e.g., oncologist or neurologist consultation, biopsy / IHC staining confirmation, further imaging scans, monitoring).

### 5. Patient Advisory & Safety Disclaimer
A concise reminder that this is an AI-assisted computational screening tool intended to support, not replace, certified medical specialist evaluation.

Format with clean markdown headers and bullet points. Ground your explanation in the clinical reference information if provided.
"""

    response = model.generate_content(prompt)
    return response.text


def validate_brain_mri(image_bytes):
    """
    Checks if the uploaded image represents a Brain MRI scan.
    """
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            "You are a medical scan intake validator. Look at this image. "
            "Is this image a brain MRI scan? Respond with only one word: 'YES' or 'NO'."
        )
        response = model.generate_content([img, prompt])
        result = response.text.strip().upper()
        return "YES" in result
    except Exception as e:
        print(f"Modality validation warning: {str(e)}")
        return True