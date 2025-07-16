import os
# Configurar variables de entorno para evitar problemas con Metal backend
os.environ["LLAMA_CPP_USE_METAL"] = "0"
os.environ["LLAMA_CPP_USE_CUBLAS"] = "0"
os.environ["LLAMA_CPP_USE_OPENBLAS"] = "0"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from chat_model import generate_response

app = FastAPI()

# Optional CORS if you're calling from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    prompt = body.get("message", "")
    if not prompt:
        return {"error": "No prompt provided"}
    # Ejecutar la generación en un threadpool para no bloquear el event loop
    response = await run_in_threadpool(generate_response, prompt)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
