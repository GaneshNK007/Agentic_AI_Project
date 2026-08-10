import streamlit as st
import requests

st.set_page_config(page_title="Medical AI Agent", layout="wide")

st.title("🧠 Medical AI Agent")
st.markdown("### 🤖 Brain Tumor + Metastatic Cancer Detection")

# =========================
# CHAT SECTION
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a medical question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        res = requests.post(
            "http://127.0.0.1:8000/agent",
            params={"query": user_input}
        )
        result = res.json()

        if "answer" in result:
            bot_response = result["answer"]
        else:
            bot_response = "⚠️ Unexpected response"

    except Exception as e:
        bot_response = f"❌ Error: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    with st.chat_message("assistant"):
        st.markdown(bot_response)

# =========================
# IMAGE UPLOAD SECTION
# =========================
st.markdown("---")
st.markdown("## 📤 Upload Medical Images")

col1, col2 = st.columns(2)

# =========================
# 🧠 BRAIN MRI
# =========================
with col1:
    st.subheader("🧠 Brain MRI")

    brain_file = st.file_uploader("Upload MRI", type=["jpg", "png"], key="brain")

    if brain_file:
        st.image(brain_file, caption="Uploaded MRI", use_container_width=True)

        with st.spinner("🧠 Analyzing Brain MRI..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/agent",
                    files={"file": (brain_file.name, brain_file, brain_file.type)},  # ✅ FIXED
                    params={"image_type": "brain"}  # ✅ FIXED
                )

                result = res.json()

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.markdown(f"### 🧠 Prediction\n**{result.get('prediction')}**")
                    st.markdown(f"### 📊 Confidence\n**{result.get('confidence')*100:.2f}%**")
                    st.markdown(f"### ⚠️ Risk Level\n**{result.get('risk')}**")
                    st.markdown(f"### 📖 Explanation\n{result.get('explanation')}")

            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

# =========================
# 🧬 METASTATIC CANCER
# =========================
with col2:
    st.subheader("🧬 Metastatic Tissue")

    cancer_file = st.file_uploader("Upload Tissue Image", type=["jpg", "png"], key="cancer")

    if cancer_file:
        st.image(cancer_file, caption="Uploaded Tissue Image", use_container_width=True)

        with st.spinner("🧬 Analyzing Tissue..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/agent",
                    files={"file": (cancer_file.name, cancer_file, cancer_file.type)},  # ✅ FIXED
                    params={"image_type": "cancer"}  # ✅ FIXED
                )

                result = res.json()

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.markdown(f"### 🧬 Prediction\n**{result.get('prediction')}**")
                    st.markdown(f"### 📊 Confidence\n**{result.get('confidence')*100:.2f}%**")
                    st.markdown(f"### ⚠️ Risk Level\n**{result.get('risk')}**")
                    st.markdown(f"### 📖 Explanation\n{result.get('explanation')}")

            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

# =========================
# DEBUG MODE
# =========================
st.markdown("---")
if st.checkbox("🧪 Show Debug Info"):
    st.write("Session State:")
    st.json(st.session_state)