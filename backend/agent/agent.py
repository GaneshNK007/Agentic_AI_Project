from backend.models.model import predict_brain
from backend.models.metaCancer import predict_cancer
from backend.llm.gemini import generate_medical_response
from backend.rag.retriever import get_answer

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType


# ==========================================
# 🔧 TOOL: Medical Explanation
# ==========================================
@tool
def medical_explanation_tool(text: str) -> str:
    """Provide medical explanations for predictions or answer health-related queries."""
    try:
        context = get_answer(text)
        return generate_medical_response(
            prediction="N/A",
            confidence=0.0,
            risk="N/A",
            user_query=text,
            context=context
        )
    except Exception as e:
        return f"⚠️ Gemini unavailable: {str(e)}"


# ==========================================
# 🤖 MAIN AGENT FUNCTION
# ==========================================
def run_agent(file=None, query=None, image_type=None, normalize=True):
    # =========================
    # 🧠 IMAGE CASE
    # =========================
    if file is not None:
        # 🔥 AGENT DECISION 1: DYNAMIC MODALITY ROUTING
        if image_type in ["brain", "brain_mri"]:
            result = predict_brain(file, normalize=normalize)
            modality = "Brain MRI Scan (Primary CNS)"
            model_key = "brain"
        elif image_type in ["cancer", "metastatic", "pcam"]:
            result = predict_cancer(file, normalize=normalize)
            modality = "Metastatic Breast Cancer Lymph Node Histopathology (CAMELYON16 / PCam)"
            model_key = "cancer"
        else:
            return {"error": f"Invalid image type '{image_type}'. Expected 'brain' or 'cancer'/'metastatic'."}

        # If model fails
        if "error" in result:
            return result

        # 🔥 AGENT DECISION 2: UNCERTAINTY INTERCEPT & ADAPTIVE TRIAGE
        uncertainty = result.get("uncertainty", 0.0)
        confidence = result.get("confidence", 0.0)
        is_uncertain = (uncertainty >= 0.65) or (confidence < 0.60)

        # =========================
        # 📚 AGENT DECISION 3: ADAPTIVE RAG CLINICAL REFERENCE RETRIEVAL
        # =========================
        try:
            if is_uncertain:
                # When uncertain, dynamically query differential diagnosis, recuts, and confirmatory IHC
                if model_key == "brain":
                    rag_query = "Glioma Meningioma differential diagnosis IDH1 GFAP IHC uncertain brain lesion"
                else:
                    rag_query = "Sentinel lymph node micrometastasis cytokeratin AE1/AE3 IHC borderline uncertain pathology"
                escalation_status = "TRIGGERED: High Classification Uncertainty — Specialist Pathologist/Radiologist Review Mandated"
            else:
                rag_query = f"{result['prediction']} {modality}"
                escalation_status = "NORMAL: Definitive High-Confidence Staging Workflow"

            clinical_reference = get_answer(rag_query)
        except Exception as e:
            clinical_reference = f"Clinical reference unavailable: {str(e)}"
            escalation_status = "DEGRADED: RAG index retrieval exception"

        # =========================
        # 🔥 AGENT DECISION 4: GUARDRAILED GEMINI GENERATION
        # =========================
        try:
            explanation = generate_medical_response(
                prediction=result["prediction"],
                confidence=result["confidence"],
                risk=f"{result['risk']} [Agent Status: {escalation_status}]",
                context=clinical_reference,
                modality=modality
            )
        except Exception as e:
            explanation = f"⚠️ Clinical report generation unavailable (Gemini error): {str(e)}"

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "uncertainty": uncertainty,
            "uncertainty_metric": result.get("uncertainty_metric", "Predictive Entropy"),
            "calibration_status": result.get("calibration_status", "Uncalibrated operational proxy"),
            "risk": result["risk"],
            "agent_escalation_status": escalation_status,
            "agent_decisions_executed": [
                f"Modality Routing -> Dispatched {model_key.upper()} CNN backbone",
                f"Uncertainty Evaluation -> Entropy = {uncertainty} (Threshold = 0.65)",
                f"Adaptive RAG -> Retrieved {'Confirmatory IHC Protocol' if is_uncertain else 'Primary Staging Protocol'}",
                "Guardrail Synthesis -> Dispatched Gemini 2.5 Flash with structured 5-part clinical template"
            ],
            "model_used": model_key,
            "modality": modality,
            "dataset_provenance": result.get("dataset_provenance", "Primary Brain Tumor MRI Benchmark"),
            "clinical_reference": clinical_reference,
            "explanation": explanation,
            "all_probabilities": result.get("all_probabilities", {})
        }

    # =========================
    # 💬 TEXT QUERY CASE
    # =========================
    elif query:
        try:
            context = get_answer(query)
            explanation = generate_medical_response(
                prediction="Consultation",
                confidence=1.0,
                risk="Informational",
                user_query=query,
                context=context
            )
            return {"answer": explanation}
        except Exception as e:
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash",
                    temperature=0.3
                )
                tools = [medical_explanation_tool]
                agent = initialize_agent(
                    tools=tools,
                    llm=llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False
                )
                response = agent.run(query)
                return {"answer": response}
            except Exception as e2:
                return {"answer": f"⚠️ LLM error: {str(e2)}"}

    return {"error": "No input provided"}