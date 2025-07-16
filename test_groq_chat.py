import os
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(
    provider="groq",
    api_key=HF_TOKEN,
)

completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "hola como estas"
        }
    ],
)

print(completion.choices[0].message.content) 