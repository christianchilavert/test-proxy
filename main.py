from fastapi import FastAPI, Request
from huggingface_hub import InferenceClient
import os
import uvicorn

app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")  # Usa variable de entorno para el token

client = InferenceClient(provider="groq", api_key=HF_TOKEN)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    message = data.get("message", "")
    history = data.get("history", [])
    messages = history + [{"role": "user", "content": message}]
    completion = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=messages,
    )
    # Devuelve solo el texto generado
    return {
        "response": completion.choices[0].message.content
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
