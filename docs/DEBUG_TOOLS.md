# 🤖 Joi Debug Tools

Sistema completo de herramientas de debug para monitorear el funcionamiento interno de Joi, el asistente de aprendizaje de inglés con WhatsApp.

## 🔧 Herramientas Disponibles

### 1. 🌐 Dashboard Web Interactivo
```bash
# Abrir en navegador
open http://localhost:8000/debug/dashboard
```

**Características:**
- 📊 Estadísticas en tiempo real
- 🏥 Estado de salud del sistema
- 👥 Lista de usuarios registrados
- 🧠 Exploración de memorias por usuario
- 💬 Historial de mensajes
- ⏰ Actividad reciente
- 🔄 Auto-refresh cada 30 segundos

### 2. 📱 Script de Línea de Comandos
```bash
# Ejecutar herramientas interactivas
python debug_tools.py

# Monitoreo rápido completo
python -c "from debug_tools import monitor_system; monitor_system()"
```

### 3. 🔗 API Endpoints de Debug

#### Estado del Sistema
```bash
# Estado de salud
curl http://localhost:8000/debug/health

# Estadísticas generales
curl http://localhost:8000/debug/stats

# Actividad reciente
curl http://localhost:8000/debug/recent-activity?limit=10
```

#### Gestión de Usuarios
```bash
# Listar usuarios
curl http://localhost:8000/debug/users

# Sesiones de un usuario
curl http://localhost:8000/debug/users/{user_id}/sessions

# Mensajes de un usuario
curl http://localhost:8000/debug/users/{user_id}/messages?limit=20

# Estadísticas de aprendizaje
curl http://localhost:8000/debug/users/{user_id}/learning-stats
```

#### Sistema de Memoria
```bash
# Ver memorias de un usuario
curl http://localhost:8000/debug/users/{user_id}/memories

# Buscar memorias específicas
curl http://localhost:8000/debug/users/{user_id}/memories?query="girlfriend"

# Probar búsqueda de memoria
curl -X POST "http://localhost:8000/debug/users/{user_id}/test-memory?query=nombre"

# Eliminar memorias de un usuario (para testing)
curl -X DELETE http://localhost:8000/debug/users/{user_id}/memories
```

#### Análisis de Sesiones
```bash
# Mensajes de una sesión específica
curl http://localhost:8000/debug/sessions/{session_id}/messages
```

## 📊 Información Monitoreada

### 🔍 Workflow Interno
El sistema registra logs detallados de cada nodo del grafo:

- **[ROUTER]** - Selección de tipo de respuesta
- **[CONTEXT]** - Inyección de contexto temporal
- **[CONVERSATION]** - Generación de respuestas
- **[IMAGE]** - Creación de imágenes
- **[AUDIO]** - Síntesis de voz
- **[SUMMARY]** - Resumen de conversaciones
- **[MEMORY_EXTRACT]** - Extracción de memorias
- **[MEMORY_INJECT]** - Inyección de memorias

### 💾 Datos Almacenados

#### Usuarios
- ID único
- Número de teléfono
- Nombre (opcional)
- Nivel actual (A1, A2, B1, etc.)
- Fecha de creación

#### Sesiones
- ID de sesión
- Usuario asociado
- Timestamp de inicio/fin
- Contexto adicional

#### Mensajes
- ID del mensaje
- Sesión asociada
- Remitente (user/agent)
- Contenido del mensaje
- Timestamp

#### Memorias
- Texto de la memoria
- Score de relevancia
- Metadatos (user_id, session_id, timestamp)
- Indexación por usuario

## 🎯 Casos de Uso Comunes

### 1. Verificar que un Usuario está Recordando Información
```python
from debug_tools import test_memory_search

# Probar si el sistema recuerda el nombre de la novia
test_memory_search("user_id", "girlfriend name")
```

### 2. Analizar Conversaciones Problemáticas
```python
from debug_tools import show_user_messages, show_user_memories

# Ver historial completo
show_user_messages("user_id", 50)

# Ver qué memorias se están formando
show_user_memories("user_id")
```

### 3. Monitorear Actividad en Tiempo Real
```python
from debug_tools import show_recent_activity

# Ver últimas 20 interacciones
show_recent_activity(20)
```

### 4. Verificar Estado del Sistema
```python
from debug_tools import check_health, show_stats

# Estado de componentes
check_health()

# Métricas generales
show_stats()
```

## 🚨 Debugging de Problemas Comunes

### Usuario No Recuerda Información
1. **Verificar memorias:** `GET /debug/users/{user_id}/memories`
2. **Probar búsqueda:** `POST /debug/users/{user_id}/test-memory`
3. **Revisar logs:** Buscar logs `[MEMORY_EXTRACT]` y `[MEMORY_INJECT]`

### Respuestas Inconsistentes
1. **Ver contexto de memoria:** Verificar `memory_context` en logs
2. **Analizar historial:** `GET /debug/users/{user_id}/messages`
3. **Verificar workflow:** Buscar logs `[ROUTER]` y `[CONVERSATION]`

### Problemas de Performance
1. **Estadísticas generales:** `GET /debug/stats`
2. **Estado de salud:** `GET /debug/health`
3. **Sesiones activas:** Verificar `active_sessions`

## 📝 Logs del Sistema

Los logs del workflow se muestran en la consola del servidor:

```
2025-06-02 21:09:00 - workflow - INFO - [ROUTER] Processing messages for user ec467509
2025-06-02 21:09:01 - workflow - INFO - [ROUTER] Selected workflow: conversation for user ec467509
2025-06-02 21:09:01 - workflow - INFO - [MEMORY_INJECT] Injecting memories for user ec467509
2025-06-02 21:09:01 - workflow - INFO - [MEMORY_INJECT] Found 3 relevant memories for user ec467509
2025-06-02 21:09:02 - workflow - INFO - [CONVERSATION] Processing conversation for user ec467509
2025-06-02 21:09:03 - workflow - INFO - [CONVERSATION] Generated response for user ec467509: 45 chars
2025-06-02 21:09:03 - workflow - INFO - [MEMORY_EXTRACT] Processing memory extraction for user ec467509
```

## 🔐 Consideraciones de Seguridad

- Los endpoints de debug están pensados para **desarrollo y testing**
- En producción, considera autenticación para estos endpoints
- Los datos de usuario mostrados deben cumplir con GDPR
- La función de eliminar memorias es útil para cumplimiento de privacidad

## 🚀 Próximas Mejoras

- [ ] Métricas de performance en tiempo real
- [ ] Alertas automáticas por problemas
- [ ] Export de datos para análisis
- [ ] Visualización de flujos de conversación
- [ ] Dashboard de métricas de aprendizaje por usuario
- [ ] Análisis de patrones de memoria
- [ ] Debugging de flujos específicos (imagen, audio)

## 🛠️ Desarrollo

Para añadir nuevos endpoints de debug:

1. Editar `src/agent/interfaces/debug/debug_endpoints.py`
2. Actualizar `debug_tools.py` para nuevas funciones CLI
3. Modificar `debug_dashboard.html` para nueva UI
4. Documentar en este README

## 📞 Ejemplo de Flujo Completo

```bash
# 1. Verificar estado del sistema
curl http://localhost:8000/debug/health

# 2. Ver usuarios activos
curl http://localhost:8000/debug/users

# 3. Analizar usuario específico
USER_ID="ec467509-23d5-4ff1-8034-4676e3a85db8"
curl http://localhost:8000/debug/users/$USER_ID/memories
curl http://localhost:8000/debug/users/$USER_ID/messages?limit=10

# 4. Probar memoria específica
curl -X POST "http://localhost:8000/debug/users/$USER_ID/test-memory?query=novia"

# 5. Ver actividad reciente
curl http://localhost:8000/debug/recent-activity
```

¡Con estas herramientas tienes visibilidad completa del funcionamiento interno de Joi! 🎉 