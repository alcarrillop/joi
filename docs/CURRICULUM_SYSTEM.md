# 🎓 Sistema de Currículo y Progresión de Aprendizaje - Joi

Sistema completo de currículo estructurado basado en el Marco Común Europeo de Referencia (CEFR) para el aprendizaje progresivo de inglés.

## 📋 Características Principales

### ✨ Currículo Estructurado
- **Niveles CEFR**: A1 (Beginner), A2 (Elementary), B1 (Intermediate), B2, C1, C2
- **Competencias Específicas**: 10+ competencias organizadas por nivel y habilidad
- **Prerrequisitos**: Sistema de dependencias entre competencias
- **Progresión Natural**: Transición automática entre niveles

### 🎯 Tipos de Competencias
- **Speaking**: Conversación e interacción oral
- **Vocabulary**: Vocabulario esencial por temas
- **Grammar**: Estructuras gramaticales progresivas
- **Listening**: Comprensión auditiva
- **Reading**: Comprensión lectora
- **Writing**: Expresión escrita

### 📊 Sistema de Evaluación
- **Evaluaciones Automáticas**: Registro de desempeño por competencia
- **Puntuaciones de Dominio**: Escala 0-1 para cada competencia
- **Feedback Detallado**: Áreas de mejora y fortalezas
- **Criterios de Transición**: Requisitos específicos para avanzar de nivel

## 🏗 Arquitectura del Sistema

### Componentes Principales

#### 1. **CurriculumManager** (`src/agent/modules/curriculum/curriculum_manager.py`)
Gestor principal que coordina todo el sistema de currículo:

```python
curriculum_manager = get_curriculum_manager()

# Inicializar progreso de usuario
progress = await curriculum_manager.initialize_user_progress(user_id, CEFRLevel.A1)

# Obtener competencia recomendada
recommended = await curriculum_manager.get_next_recommended_competency(user_id)

# Registrar evaluación
assessment = AssessmentResult(user_id, competency_id, skill_type, score, max_score, ...)
await curriculum_manager.record_assessment(assessment)

# Obtener estadísticas
stats = await curriculum_manager.get_learning_statistics(user_id)
```

#### 2. **Modelos de Datos** (`src/agent/modules/curriculum/models.py`)
Definiciones de todas las estructuras de datos:

- `CEFRLevel`: Niveles del marco europeo
- `Competency`: Competencias específicas del currículo
- `UserProgress`: Progreso de cada usuario
- `AssessmentResult`: Resultados de evaluaciones
- `LevelTransitionCriteria`: Criterios para avanzar

#### 3. **Datos del Currículo** (`src/agent/modules/curriculum/curriculum_data.py`)
Definición completa del currículo estructurado:

```python
# Ejemplo de competencia A1
Competency(
    id="a1_introductions",
    name="Introductions and Personal Information",
    level=CEFRLevel.A1,
    skill_type=SkillType.SPEAKING,
    key_vocabulary=["name", "age", "from", "live", ...],
    grammar_points=["Present simple 'be'", "Wh-questions", ...],
    estimated_hours=3
)
```

### Base de Datos

#### Tablas Principales
- `user_progress`: Progreso de cada usuario
- `assessments`: Resultados de evaluaciones
- `learning_goals`: Metas personalizadas
- `practice_activities`: Actividades completadas
- `vocabulary_progress`: Seguimiento de vocabulario
- `grammar_errors`: Errores identificados

#### Vistas Analíticas
- `user_progress_summary`: Resumen de progreso
- `vocabulary_stats`: Estadísticas de vocabulario
- `grammar_error_analysis`: Análisis de errores

## 📚 Competencias por Nivel

### Nivel A1 (Beginner) - 5 Competencias
1. **Introductions and Personal Information** (Speaking)
   - Presentarse y dar información básica
   - Vocabulario: name, age, from, live, hello, goodbye...
   - Gramática: Present simple 'be', pronombres personales

2. **Essential Daily Vocabulary** (Vocabulary)
   - 200+ palabras fundamentales
   - Vocabulario: eat, drink, sleep, work, good, bad...
   - Gramática: Artículos (a, an, the), singular/plural

3. **Present Simple Tense** (Grammar)
   - Presente simple para rutinas diarias
   - Prerrequisito: Essential Daily Vocabulary
   - Gramática: Do/Does, adverbios de frecuencia

4. **Family and Relationships** (Vocabulary)
   - Miembros de la familia y relaciones
   - Vocabulario: mother, father, sister, brother...
   - Gramática: Adjetivos posesivos, have/has

5. **Numbers and Time** (Vocabulary)
   - Números 1-100, fechas y tiempo
   - Vocabulario: números, días, meses, morning...
   - Gramática: Preposiciones de tiempo

### Nivel A2 (Elementary) - 3 Competencias
1. **Past Simple Tense** (Grammar)
   - Hablar sobre eventos pasados
   - Prerrequisito: Present Simple Tense
   - Gramática: Verbos regulares/irregulares, was/were

2. **Food and Dining** (Vocabulary)
   - Ordenar comida y expresar preferencias
   - Vocabulario: breakfast, lunch, chicken, delicious...
   - Gramática: Would like, contables/incontables

3. **Travel and Directions** (Vocabulary)
   - Viajar y pedir direcciones
   - Vocabulario: airport, hotel, left, right, near...
   - Gramática: Preposiciones de lugar, imperativo

### Nivel B1 (Intermediate) - 2 Competencias
1. **Future Plans and Predictions** (Grammar)
   - Planes futuros y predicciones
   - Prerrequisito: Past Simple Tense
   - Gramática: Will vs going to, presente continuo futuro

2. **Work and Career** (Vocabulary)
   - Trabajo y objetivos profesionales
   - Vocabulario: job, career, salary, manager...
   - Gramática: Presente perfecto, verbos modales

## 🎯 Criterios de Transición

### A1 → A2
- **Competencias completadas**: 4 de 5
- **Puntuación promedio**: ≥75%
- **Competencias core**: introductions, basic_vocabulary, present_simple
- **Tiempo mínimo**: 14 días

### A2 → B1
- **Competencias completadas**: 3 de 3
- **Puntuación promedio**: ≥80%
- **Competencias core**: past_simple, food_vocabulary
- **Tiempo mínimo**: 21 días

### B1 → B2
- **Competencias completadas**: 2 de 2
- **Puntuación promedio**: ≥85%
- **Competencias core**: future_simple, work_vocabulary
- **Tiempo mínimo**: 30 días

## 🔧 API Debug Endpoints

### Estadísticas del Currículo
```bash
# Estadísticas generales
GET /debug/stats

# Progreso de usuario específico
GET /debug/users/{user_id}/curriculum-progress

# Evaluaciones de usuario
GET /debug/users/{user_id}/assessments

# Todas las competencias
GET /debug/curriculum/competencies
```

### Ejemplo de Respuesta - Progreso de Usuario
```json
{
    "user_id": "uuid",
    "current_level": "A1",
    "completed_competencies": ["a1_introductions"],
    "in_progress_competencies": [],
    "mastery_scores": {
        "a1_introductions": 0.85
    },
    "available_competencies": [...],
    "recommended_competency": {
        "id": "a1_family_vocabulary",
        "name": "Family and Relationships",
        "skill_type": "vocabulary",
        "estimated_hours": 3
    }
}
```

## 📊 Monitoreo y Análisis

### Dashboard de Debug
- **Estadísticas en tiempo real**: Progreso de todos los usuarios
- **Visualización de competencias**: Estado por usuario
- **Análisis de evaluaciones**: Tendencias de aprendizaje
- **Seguimiento de transiciones**: Avance de niveles

### Métricas Clave
- Total de evaluaciones registradas
- Competencias completadas por usuario
- Tiempo promedio por nivel
- Tasas de éxito por competencia
- Análisis de errores comunes

## 🚀 Integración Futura

### Próximas Características
- **Evaluación automática en conversaciones**
- **Detección de gramática y vocabulario**
- **Recomendaciones personalizadas de práctica**
- **Sistema de repetición espaciada**
- **Gamificación y logros**

### Extensiones Planificadas
- **Más niveles**: B2, C1, C2
- **Más competencias por nivel**
- **Especialización por dominios**: Business English, Academic English
- **Integración con ejercicios interactivos**
- **Análisis de voz y pronunciación**

## 📈 Beneficios del Sistema

1. **Progresión Estructurada**: Aprendizaje ordenado y sin lagunas
2. **Personalización**: Recomendaciones basadas en progreso individual
3. **Feedback Inmediato**: Evaluación y retroalimentación continua
4. **Motivación**: Visualización clara del progreso
5. **Adaptabilidad**: Sistema flexible que se ajusta al ritmo del usuario

---

## 🛠 Instalación y Configuración

1. **Crear tablas de base de datos**:
   ```bash
   python create_curriculum_tables.py
   ```

2. **Probar el sistema**:
   ```bash
   python test_curriculum.py
   ```

3. **Acceder al dashboard**:
   ```
   http://localhost:8000/debug/dashboard
   ```

El sistema de currículo está completamente integrado y listo para proporcionar una experiencia de aprendizaje estructurada y progresiva para todos los usuarios de Joi. 🎉 