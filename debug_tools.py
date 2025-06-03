#!/usr/bin/env python3
"""
Herramientas de debug para Joi - AI Language Learning Assistant
"""
import requests
import json
import webbrowser
from datetime import datetime

API_BASE = "http://localhost:8000/debug"

def print_json(data):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2, ensure_ascii=False))

def check_health():
    """Verificar estado de salud del sistema."""
    print("🏥 Verificando estado de salud del sistema...")
    try:
        response = requests.get(f"{API_BASE}/health")
        health = response.json()
        print_json(health)
        return health
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def show_stats():
    """Mostrar estadísticas del sistema."""
    print("📊 Estadísticas del sistema:")
    try:
        response = requests.get(f"{API_BASE}/stats")
        stats = response.json()
        print_json(stats)
        return stats
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def list_users():
    """Listar todos los usuarios."""
    print("👥 Usuarios registrados:")
    try:
        response = requests.get(f"{API_BASE}/users")
        users = response.json()
        for user in users:
            print(f"📱 {user['phone_number']} - {user['current_level']} (ID: {user['id']})")
        return users
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def show_user_memories(user_id):
    """Mostrar memorias de un usuario específico."""
    print(f"🧠 Memorias del usuario {user_id}:")
    try:
        response = requests.get(f"{API_BASE}/users/{user_id}/memories")
        memories = response.json()
        for memory in memories:
            print(f"  - {memory['text']} (score: {memory['score']:.3f})")
        return memories
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def show_user_messages(user_id, limit=10):
    """Mostrar mensajes de un usuario."""
    print(f"💬 Últimos {limit} mensajes del usuario {user_id}:")
    try:
        response = requests.get(f"{API_BASE}/users/{user_id}/messages?limit={limit}")
        messages = response.json()
        for msg in messages:
            timestamp = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
            sender_icon = "👤" if msg['sender'] == 'user' else "🤖"
            print(f"  {sender_icon} [{timestamp.strftime('%H:%M:%S')}] {msg['message']}")
        return messages
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_memory_search(user_id, query):
    """Probar búsqueda de memoria."""
    print(f"🔍 Probando búsqueda de memoria para '{query}':")
    try:
        response = requests.post(f"{API_BASE}/users/{user_id}/test-memory?query={query}")
        result = response.json()
        print(f"Memorias encontradas: {result['memories_found']}")
        for memory in result['raw_memories']:
            print(f"  - {memory}")
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def show_recent_activity(limit=10):
    """Mostrar actividad reciente."""
    print(f"⏰ Últimas {limit} interacciones:")
    try:
        response = requests.get(f"{API_BASE}/recent-activity?limit={limit}")
        activity = response.json()
        for item in activity:
            timestamp = datetime.fromisoformat(item['timestamp'].replace('+00:00', ''))
            sender_icon = "👤" if item['sender'] == 'user' else "🤖"
            print(f"  {sender_icon} [{timestamp.strftime('%H:%M:%S')}] {item['phone_number']}: {item['message'][:80]}...")
        return activity
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def open_dashboard():
    """Abrir el dashboard web en el navegador."""
    print("🌐 Abriendo dashboard web...")
    webbrowser.open(f"{API_BASE}/dashboard")

def monitor_system():
    """Ejecutar un monitoreo completo del sistema."""
    print("=" * 50)
    print("🤖 JOI DEBUG TOOLS - Sistema de Monitoreo")
    print("=" * 50)
    
    # Check health
    health = check_health()
    print("\n" + "-" * 30)
    
    # Show stats
    stats = show_stats()
    print("\n" + "-" * 30)
    
    # List users
    users = list_users()
    print("\n" + "-" * 30)
    
    # Show recent activity
    show_recent_activity(5)
    
    if users:
        print("\n" + "-" * 30)
        print("🔍 Análisis detallado del primer usuario:")
        first_user = users[0]
        user_id = first_user['id']
        
        show_user_memories(user_id)
        print()
        show_user_messages(user_id, 5)

def main():
    """Función principal con menú interactivo."""
    while True:
        print("\n" + "=" * 50)
        print("🤖 JOI DEBUG TOOLS")
        print("=" * 50)
        print("1. 🏥 Verificar estado de salud")
        print("2. 📊 Mostrar estadísticas")
        print("3. 👥 Listar usuarios")
        print("4. 🧠 Ver memorias de un usuario")
        print("5. 💬 Ver mensajes de un usuario")
        print("6. 🔍 Probar búsqueda de memoria")
        print("7. ⏰ Ver actividad reciente")
        print("8. 🌐 Abrir dashboard web")
        print("9. 📋 Monitoreo completo")
        print("0. 🚪 Salir")
        
        try:
            choice = input("\nSelecciona una opción: ").strip()
            
            if choice == "0":
                print("👋 ¡Hasta luego!")
                break
            elif choice == "1":
                check_health()
            elif choice == "2":
                show_stats()
            elif choice == "3":
                list_users()
            elif choice == "4":
                user_id = input("ID del usuario: ").strip()
                show_user_memories(user_id)
            elif choice == "5":
                user_id = input("ID del usuario: ").strip()
                limit = int(input("Número de mensajes (default 10): ") or "10")
                show_user_messages(user_id, limit)
            elif choice == "6":
                user_id = input("ID del usuario: ").strip()
                query = input("Consulta de búsqueda: ").strip()
                test_memory_search(user_id, query)
            elif choice == "7":
                limit = int(input("Número de actividades (default 10): ") or "10")
                show_recent_activity(limit)
            elif choice == "8":
                open_dashboard()
            elif choice == "9":
                monitor_system()
            else:
                print("❌ Opción no válida")
                
        except KeyboardInterrupt:
            print("\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 