# Sistema de Evaluación Automática - Step 6

## 🎯 Descripción General

El sistema de evaluación automática analiza las conversaciones de los usuarios en tiempo real para:
- Detectar su nivel de inglés (A1-B2)
- Evaluar habilidades específicas (vocabulario, gramática, fluidez)
- Proporcionar feedback personalizado
- Integrar con el sistema de currículo
- Detectar errores comunes

## 📊 Componentes Principales

### 1. Analizadores de Texto (`src/agent/modules/assessment/analyzers.py`)

#### VocabularyAnalyzer
- **Función**: Analiza el vocabulario usado por el usuario
- **Características**:
  - Clasifica palabras por nivel CEFR (A1-B2)
  - Identifica vocabulario avanzado vs básico
  - Calcula complejidad y apropiación del vocabulario
  - Filtra palabras funcionales irrelevantes

#### GrammarAnalyzer
- **Función**: Detecta errores gramaticales y analiza estructuras
- **Características**:
  - Detección de errores con patrones regex
  - Identificación de tiempos verbales usados
  - Análisis de estructuras oracionales
  - Cálculo de precisión gramatical

#### FluencyAnalyzer
- **Función**: Evalúa la fluidez y naturalidad de la comunicación
- **Características**:
  - Análisis de coherencia del mensaje
  - Evaluación de claridad de expresión
  - Medición de fluidez natural
  - Consideración de tiempo de respuesta

### 2. Gestor de Evaluación (`src/agent/modules/assessment/assessment_manager.py`)

#### AssessmentManager
- **Función**: Coordina todos los análisis y toma decisiones
- **Características**:
  - Evaluación automática durante conversaciones
  - Detección de nivel agregada
  - Seguimiento de progresión de habilidades
  - Integración con sistema de currículo
  - Cache de evaluaciones recientes

### 3. Modelos de Datos (`src/agent/modules/assessment/models.py`)

#### Estructuras Principales
- `ConversationAssessment`: Evaluación completa de un mensaje
- `LevelDetectionResult`: Resultado de detección de nivel
- `SkillProgression`: Progresión de habilidades específicas
- `VocabularyAnalysis`, `GrammarAnalysis`, `FluencyAnalysis`: Análisis detallados

## 🗄️ Base de Datos

### Tablas Creadas

#### conversation_assessments
```sql
- id, user_id, session_id, message_id
- timestamp, user_message
- overall_level, confidence_score, skills_scores
- strengths[], areas_for_improvement[]
- competency_evidence (JSONB)
- vocab_level, grammar_level
- vocab_complexity, grammar_accuracy, fluency_score
```

#### level_detections
```sql
- id, user_id, detected_level, confidence
- evidence (JSONB), recommendation
- should_advance, should_review
- assessment_count, detection_date, is_active
```

#### skill_progressions
```sql
- id, user_id, skill, current_score
- trend, level_estimate, next_milestone
- scores_history (JSONB), last_updated
```

#### detected_errors
```sql
- id, user_id, assessment_id
- error_type, original_text, corrected_text
- explanation, error_position (JSONB)
- severity, detected_at
```

#### assessment_configs
```sql
- user_id (PK), configuración personalizada por usuario
- min_words_for_assessment, assessment_frequency
- confidence_threshold, level_change_threshold
- pesos para diferentes aspectos de evaluación
```

### Vistas de Análisis

#### assessment_summary
- Resumen de evaluaciones por usuario (últimos 30 días)
- Estadísticas agregadas: nivel más común, confianza promedio
- Puntuaciones promedio por área

#### skill_overview
- Vista general de progresión de habilidades
- Puntuación actual, tendencia, estimación de nivel
- Próximos hitos por habilidad

#### error_patterns
- Patrones de errores más comunes
- Severidad promedio, frecuencia, afectados
- Agrupado por tipo de error

## 🔧 API de Debug

### Endpoints Principales

#### GET `/debug/users/{user_id}/assessment-history`
- Historial de evaluaciones automáticas
- Parámetros: `limit` (default: 20)
- Incluye análisis detallados por mensaje

#### GET `/debug/users/{user_id}/level-detection`
- Detección de nivel más reciente
- Genera detección en tiempo real si no existe
- Incluye recomendaciones de avance/revisión

#### GET `/debug/users/{user_id}/skill-progression`
- Progresión de todas las habilidades
- Puntuación actual, tendencia, próximos hitos
- Historial de puntuaciones con timestamps

#### POST `/debug/users/{user_id}/assess-message`
- Evaluación inmediata de un mensaje específico
- Fuerza evaluación (ignora filtros de frecuencia)
- Retorna análisis completo en tiempo real

#### GET `/debug/assessment/summary`
- Resumen general del sistema
- Estadísticas globales, distribución de niveles
- Tendencias de progresión, habilidades más evaluadas

#### GET `/debug/assessment/error-patterns`
- Patrones de errores detectados
- Parámetros: `user_id` (opcional), `days` (default: 30)
- Análisis global o por usuario específico

## ⚙️ Configuración

### Parámetros del Sistema
```python
class AutoAssessmentConfig:
    min_words_for_assessment: int = 5        # Mínimo de palabras para evaluar
    assessment_frequency: int = 3            # Evaluar cada N mensajes
    confidence_threshold: float = 0.7        # Umbral de confianza
    level_change_threshold: float = 0.8      # Umbral para cambio de nivel
    
    # Pesos para cálculo de nivel general
    grammar_weight: float = 0.3
    vocabulary_weight: float = 0.3
    fluency_weight: float = 0.2
    accuracy_weight: float = 0.2
```

## 🔄 Flujo de Evaluación

### 1. Evaluación Automática
```
Usuario envía mensaje → 
Filtros (longitud, frecuencia) → 
Análisis paralelo (vocab, gramática, fluidez) → 
Cálculo de nivel general → 
Generación de feedback → 
Almacenamiento en BD → 
Actualización de currículo
```

### 2. Detección de Nivel
```
Obtener evaluaciones recientes (7 días) → 
Analizar tendencias y consistencia → 
Calcular evidencia por habilidad → 
Determinar nivel más probable → 
Generar recomendaciones → 
Actualizar detección activa
```

### 3. Progresión de Habilidades
```
Extraer puntuaciones históricas → 
Calcular tendencias (mejorando/estable/declinando) → 
Estimar nivel por habilidad → 
Definir próximos hitos → 
Actualizar base de datos
```

## 📈 Métricas y Análisis

### Niveles CEFR Soportados
- **A1**: Vocabulario básico, estructuras simples
- **A2**: Vocabulario cotidiano, tiempos pasado/futuro
- **B1**: Vocabulario académico, estructuras complejas
- **B2**: Vocabulario avanzado, uso natural de gramática

### Habilidades Evaluadas
- **Vocabulary**: Complejidad y apropiación del vocabulario
- **Grammar**: Precisión gramatical y uso de estructuras
- **Fluency**: Fluidez natural y coherencia
- **Accuracy**: Precisión general en el uso del idioma
- **Complexity**: Complejidad sintáctica y léxica
- **Coherence**: Coherencia y cohesión del discurso

### Tipos de Errores Detectados
- **subject_verb_agreement**: Concordancia sujeto-verbo
- **verb_tense**: Uso incorrecto de tiempos verbales
- **article_usage**: Uso incorrecto de artículos (a/an)

## 🧪 Pruebas y Validación

### Test Suite (`test_assessment_system.py`)
- Pruebas con mensajes de diferentes niveles de complejidad
- Validación de detección de nivel
- Verificación de progresión de habilidades
- Pruebas de componentes individuales
- Detección de errores específicos

### Resultados de Pruebas
```
✓ Evaluaciones realizadas: 4 casos de prueba
✓ Detección de nivel completada
✓ Progresión de habilidades analizada
✓ Componentes de análisis verificados
✓ Detección de errores funcionando
```

## 🔮 Integración con Currículo

### Evidencia de Competencias
- El sistema identifica automáticamente evidencia de competencias específicas
- Mapea vocabulario usado con vocabulario clave de competencias
- Correlaciona habilidades con tipos de competencias del currículo
- Genera evaluaciones automáticas cuando hay evidencia suficiente (>70%)

### Feedback Personalizado
- **Fortalezas identificadas**: Áreas donde el usuario demuestra competencia
- **Áreas de mejora**: Aspectos específicos que necesitan trabajo
- **Recomendaciones**: Sugerencias basadas en el nivel detectado y progresión

## 📋 Instalación y Uso

### 1. Crear Tablas
```bash
python create_assessment_tables.py
```

### 2. Ejecutar Pruebas
```bash
python test_assessment_system.py
```

### 3. Acceder al Dashboard
```
http://localhost:8000/debug/dashboard
```

### 4. Endpoints API
```
http://localhost:8000/debug/assessment/summary
http://localhost:8000/debug/users/{user_id}/assessment-history
```

## 🎉 Características Destacadas

### ✅ Evaluación en Tiempo Real
- Análisis automático durante conversaciones naturales
- Sin interrumpir el flujo de la conversación
- Feedback inmediato disponible para debug

### ✅ Análisis Multicapa
- Vocabulario: Nivel CEFR y complejidad
- Gramática: Errores y estructuras usadas
- Fluidez: Coherencia y naturalidad

### ✅ Sistema Adaptativo
- Configuración personalizable por usuario
- Filtros inteligentes para evitar sobre-evaluación
- Cache para rendimiento optimizado

### ✅ Integración Completa
- Conexión automática con sistema de currículo
- Actualización de progreso en tiempo real
- Evidencia objetiva para evaluaciones

### ✅ Herramientas de Debug Avanzadas
- Evaluación forzada para testing
- Análisis detallado de componentes
- Métricas y tendencias históricas
- Patrones de errores identificados

El sistema está completamente funcional y listo para el siguiente paso del plan. 