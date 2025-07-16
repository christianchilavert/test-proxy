# chat_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from config import MODEL_NAME

# Cargar el tokenizer y el modelo Qwen2.5-14B
print(f"Cargando modelo: {MODEL_NAME}")
# Carpeta temporal para offload
OFFLOAD_FOLDER = "./hf_offload"
os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16
)

# Definir pad_token_id si no existe
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.pad_token = tokenizer.unk_token

def generate_response(prompt, max_length=128):
    # Tokenizar el input
    inputs = tokenizer(prompt + tokenizer.eos_token, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Generar respuesta
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decodificar la respuesta
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extraer solo la parte generada (sin el prompt original)
    if prompt in response:
        response = response.replace(prompt, "").strip()
    
    return response.strip()
