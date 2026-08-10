from fastapi import FastAPI, UploadFile, File
from backend.agent.agent import run_agent

app = FastAPI()

# =========================
# 🤖 MAIN AGENT ENDPOINT
# =========================
@app.post("/agent")
async def agent(
    file: UploadFile = File(None),
    query: str = None,
    image_type: str = None   # 🔥 IMPORTANT
):
    return run_agent(file=file, query=query, image_type=image_type)