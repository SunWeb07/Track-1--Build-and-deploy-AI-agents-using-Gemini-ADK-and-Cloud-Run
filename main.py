from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
from google.oauth2 import service_account
import google.auth
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load credentials: use service account file locally, ADC on Cloud Run
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if credentials_path or os.path.exists("credits.json"):
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path or "credits.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
else:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

# Initialize Google Gen AI client with Vertex AI backend
client = genai.Client(
    vertexai=True,
    project="encoded-zoo-491423-h2",
    location="us-central1",
    credentials=credentials
)

class Request(BaseModel):
    text: str

@app.get("/")
def root():
    return FileResponse("index.html")

@app.post("/explain")
def explain(req: Request):
    prompt = f"""
    Explain the following concept in very simple English (like teaching a beginner).

    Also provide a simple diagram using arrows or flow format.

    Concept: {req.text}

    Return response strictly in JSON:
    {{
      "simple_explanation": "...",
      "diagram": "..."
    }}
    """

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt
    )

    try:
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except:
        return {"result": response.text}