#!/usr/bin/env python3

import os
import sys
import time

# Configurar variables de entorno para evitar problemas con Metal backend
os.environ["LLAMA_CPP_USE_METAL"] = "0"
os.environ["LLAMA_CPP_USE_CUBLAS"] = "0"
os.environ["LLAMA_CPP_USE_OPENBLAS"] = "0"

def test_model():
    print("🧪 Iniciando prueba del modelo Qwen2.5-3B...")
    print("=" * 50)
    
    try:
        # Importar las funciones del modelo
        from chat_model import generate_response
        from config import MODEL_NAME
        
        print(f"✅ Modelo configurado: {MODEL_NAME}")
        
        # Test simple
        print("\n📝 Test: Respuesta simple")
        test_prompt = "Puedes traducir este texto al inglés? Estoy contento porque es verano  "
        print(f"Prompt: {test_prompt}")
        
        start_time = time.time()
        response = generate_response(test_prompt, max_length=128)
        end_time = time.time()
        
        print(f"Respuesta: {response}")
        print(f"Tiempo de respuesta: {end_time - start_time:.2f} segundos")
        
        print("\n✅ ¡Test completado exitosamente!")
        print("🎉 El modelo está funcionando correctamente")
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Asegúrate de que todas las dependencias estén instaladas:")
        print("   pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {e}")
        print("💡 Verifica que el modelo esté descargado correctamente")

if __name__ == "__main__":
    test_model() 