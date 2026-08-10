from backend.models.model import predict_brain
from backend.models.metaCancer import predict_cancer
from backend.llm.gemini import generate_medical_response

from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType


# =========================
# 🔧 TOOL: Medical Explanation
# =========================
@tool
def medical_explanation_tool(text: str) -> str:
    """Provide medical explanations for predictions or answer health-related queries."""
    try:
        return generate_medical_response(
            prediction="N/A",
            confidence=0.0,
            risk="N/A",
            user_query=text
        )
    except Exception as e:
        return f"⚠️ Gemini unavailable: {str(e)}"


# =========================
# 🤖 MAIN AGENT FUNCTION
# =========================
def run_agent(file=None, query=None, image_type=None):

    # =========================
    # 🧠 IMAGE CASE
    # =========================
    if file is not None:

        # 🔥 ROUTING LOGIC
        if image_type == "brain":
            result = predict_brain(file)

        elif image_type == "cancer":
            result = predict_cancer(file)

        else:
            return {"error": "Invalid image type"}

        # If model fails
        if "error" in result:
            return result

        # =========================
        # 🔥 SAFE GEMINI CALL
        # =========================
        try:
            explanation = generate_medical_response(
                prediction=result["prediction"],
                confidence=result["confidence"],
                risk=result["risk"]
            )
        except Exception as e:
            explanation = f"⚠️ Explanation unavailable (Gemini error): {str(e)}"

        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "risk": result["risk"],
            "model_used": image_type,
            "explanation": explanation
        }

    # =========================
    # 💬 TEXT QUERY CASE
    # =========================
    elif query:

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
                verbose=True
            )

            response = agent.run(query)

        except Exception as e:
            response = f"⚠️ LLM error: {str(e)}"

        return {"answer": response}

    return {"error": "No input provided"}