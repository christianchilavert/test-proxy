import gradio as gr
from chat_model import generate_response

def chat_interface(message, history_json):
    # history_json es una lista de pares de strings o un string vacío
    if not history_json:
        history = []
    else:
        history = history_json
    # Limpieza básica del historial
    cleaned = []
    for item in history:
        if isinstance(item, list) and len(item) == 2:
            cleaned.append([str(item[0]), str(item[1])])
    response = generate_response(message)
    cleaned.append([str(message), str(response)])
    return cleaned, cleaned

iface = gr.Interface(
    fn=chat_interface,
    inputs=[
        gr.Textbox(label="Tu mensaje"),
        gr.JSON(label="Historial", visible=False)
    ],
    outputs=[
        gr.JSON(label="Historial actualizado"),
        gr.JSON(label="Historial para siguiente turno", visible=False)
    ],
    title="Chatbot Qwen1.5-1.8B"
)

iface.launch() 
