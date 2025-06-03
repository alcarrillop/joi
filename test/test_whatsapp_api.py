"""
Script para probar el endpoint de WhatsApp simulando mensajes
"""
import asyncio
import httpx
import json
from datetime import datetime

# Configuración del servidor local
SERVER_URL = "http://localhost:8000"
WEBHOOK_URL = f"{SERVER_URL}/whatsapp_response"

def create_whatsapp_message(from_number: str, message_text: str, message_type: str = "text"):
    """Crear un payload simulado de mensaje de WhatsApp"""
    timestamp = str(int(datetime.now().timestamp()))
    
    message_data = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "ENTRY_ID",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {
                        "display_phone_number": "15550123456",
                        "phone_number_id": "PHONE_NUMBER_ID"
                    },
                    "contacts": [{
                        "profile": {"name": "Test User"},
                        "wa_id": from_number
                    }],
                    "messages": [{
                        "from": from_number,
                        "id": f"wamid.test_{timestamp}",
                        "timestamp": timestamp,
                        "type": message_type
                    }]
                },
                "field": "messages"
            }]
        }]
    }
    
    if message_type == "text":
        message_data["entry"][0]["changes"][0]["value"]["messages"][0]["text"] = {
            "body": message_text
        }
    
    return message_data

async def send_test_message(from_number: str, message_text: str):
    """Enviar un mensaje de prueba al webhook"""
    payload = create_whatsapp_message(from_number, message_text)
    
    print(f"📤 Enviando mensaje desde {from_number}: '{message_text}'")
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                WEBHOOK_URL,
                json=payload,
                timeout=30.0
            )
            
            print(f"📨 Respuesta HTTP: {response.status_code}")
            print(f"   Contenido: {response.text}")
            
            if response.status_code == 200:
                print("✅ Mensaje procesado exitosamente")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Error de conexión: {e}")

async def test_conversation():
    """Probar una conversación completa"""
    print("🧪 Iniciando prueba de conversación completa...")
    
    # Usuario de prueba
    test_phone = "+1234567890"
    
    # Serie de mensajes de prueba
    messages = [
        "Hello, I want to learn English",
        "My name is Alice and I'm from Mexico",
        "I love reading books and playing guitar",
        "What's my name?",
        "What do I like to do?",
        "Help me practice English conversation"
    ]
    
    for i, message in enumerate(messages, 1):
        print(f"\n--- Mensaje {i}/{len(messages)} ---")
        await send_test_message(test_phone, message)
        
        # Esperar un poco entre mensajes
        print("⏳ Esperando 3 segundos...")
        await asyncio.sleep(3)
    
    print("\n✅ Prueba de conversación completada!")

async def test_health_check():
    """Probar que el servidor está funcionando"""
    print("🏥 Probando que el servidor esté activo...")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{SERVER_URL}/docs")
            if response.status_code == 200:
                print("✅ Servidor FastAPI activo")
                print(f"   Documentación disponible en: {SERVER_URL}/docs")
            else:
                print(f"⚠️  Servidor responde pero con código: {response.status_code}")
        except Exception as e:
            print(f"❌ Servidor no disponible: {e}")
            print("   Asegúrate de que el servidor esté ejecutándose con:")
            print("   uv run uvicorn src.agent.interfaces.whatsapp.webhook_endpoint:app --host 0.0.0.0 --port 8000 --reload")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "health":
            asyncio.run(test_health_check())
        elif sys.argv[1] == "single":
            phone = sys.argv[2] if len(sys.argv) > 2 else "+1234567890"
            message = sys.argv[3] if len(sys.argv) > 3 else "Hello, test message"
            asyncio.run(send_test_message(phone, message))
        elif sys.argv[1] == "conversation":
            asyncio.run(test_conversation())
    else:
        print("🎯 Opciones de prueba:")
        print("  python test_whatsapp_api.py health          - Probar que el servidor esté activo")
        print("  python test_whatsapp_api.py single [phone] [message] - Enviar un mensaje individual")
        print("  python test_whatsapp_api.py conversation    - Probar conversación completa")
        print()
        print("🚀 Ejecutando prueba de salud por defecto...")
        asyncio.run(test_health_check()) 