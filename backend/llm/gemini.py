import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key="GEMINI_API_KEY")


def generate_medical_response(prediction, confidence, risk, user_query=None):

    model = genai.GenerativeModel("gemini-2.5-flash")

    if user_query:
        prompt = f"""
        You are a medical AI assistant.

        Answer the following query in a clear and simple way:
        {user_query}
        """

    else:
        prompt = f"""
        You are a medical AI assistant.

        A brain MRI scan was analyzed.

        Prediction: {prediction}
        Confidence: {confidence}
        Risk Level: {risk}

        Explain:
        - What this condition means
        - Possible symptoms
        - Whether it is serious
        - What the patient should do next

        Keep it simple and medically accurate.
        """

    response = model.generate_content(prompt)

    return response.text