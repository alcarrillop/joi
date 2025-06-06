# Scripts Directory

Scripts y utilidades para el sistema Joi - AI English Learning Assistant.

## 🚀 Production Scripts

### `start.sh`
**Boot script para producción (Railway/Docker)**
- Configura puerto vía variable de entorno `PORT`
- Establece PYTHONPATH correcto
- Inicia servidor uvicorn con configuración optimizada

```bash
./scripts/start.sh
# O con puerto específico:
PORT=3000 ./scripts/start.sh
```

### `setup.sh`
**Script de configuración completa del entorno de desarrollo**
- Verifica versión de Python (>=3.12)
- Instala y configura uv
- Instala dependencias del proyecto
- Verifica configuración de .env
- Valida imports y configuración

```bash
./scripts/setup.sh
```

## 🗄️ Database Scripts

### `init_database.py`
**Inicialización inteligente de base de datos** ⭐ **NUEVO**
- **Detecta automáticamente** qué tablas faltan
- **Crea solo** las tablas necesarias
- **Seguro** para desarrollo y producción
- **Reemplaza** scripts de verificación/reparación

```bash
PYTHONPATH=src uv run python scripts/init_database.py
```

### `init_db.sql`
**Schema SQL base** (usado por `init_database.py`)
- Define todas las tablas esenciales
- Incluye funciones PostgreSQL para vocabulario
- Usa `IF NOT EXISTS` para seguridad

```bash
# Aplicación manual (si prefieres SQL directo)
psql $DATABASE_URL -f scripts/init_db.sql
```

## 🔧 Utility Scripts

### `update_database_schema.py`
**Verificación de schema de base de datos**
- Verifica existencia de tablas esenciales
- Muestra estructura de tablas
- Reporta conteos de registros

```bash
PYTHONPATH=src uv run python scripts/update_database_schema.py
```

### `fix_checkpoint_schema.py`
**Reparación de schema de LangGraph checkpoints**
- Resuelve problemas de "task_path column already exists"
- Recrea tablas de checkpoint con schema correcto
- Útil después de actualizaciones de LangGraph

```bash
PYTHONPATH=src uv run python scripts/fix_checkpoint_schema.py
```

## 📁 Migration Scripts (Historical)

Estos scripts SQL han sido aplicados y son mantenidos por referencia histórica:

- `add_message_type_column.sql` - Añade soporte para tipos de mensaje
- `create_word_stats_table.sql` - Crea tabla relacional de vocabulario
- `remove_learning_stats_table.sql` - Elimina tabla obsoleta

## ⚡ Quick Start

### Para Desarrollo Nuevo:
```bash
# 1. Configurar entorno
./scripts/setup.sh

# 2. Inicializar base de datos (inteligente)
PYTHONPATH=src uv run python scripts/init_database.py

# 3. Ejecutar tests
make test

# 4. Iniciar desarrollo
langgraph dev
```

### Para Producción:
```bash
# Base de datos se inicializa automáticamente en Railway
# Solo necesitas:
./scripts/start.sh
```

## 📋 Requirements

- **Python**: >=3.12
- **uv**: Package manager
- **PostgreSQL**: Base de datos
- **Variables de entorno**: DATABASE_URL, API keys

## 🛠️ Troubleshooting

### Problemas de Base de Datos
```bash
# Reinicializar tablas faltantes
PYTHONPATH=src uv run python scripts/init_database.py
```

### Reconfigurar Entorno
```bash
./scripts/setup.sh
```

## 📝 Notas

- **Filosofía**: Solo scripts necesarios, sin redundancia
- **Inteligente**: `init_database.py` detecta qué crear
- **Seguro**: Funciona en desarrollo y producción
- **Mantenible**: Menos scripts = menos mantenimiento 