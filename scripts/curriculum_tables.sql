-- Tablas para el sistema de currículo y progresión de aprendizaje

-- Tabla para el progreso del usuario en el currículo
CREATE TABLE IF NOT EXISTS user_progress (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    current_level VARCHAR(2) NOT NULL DEFAULT 'A1',
    level_start_date TIMESTAMP,
    completed_competencies TEXT[] DEFAULT '{}',
    in_progress_competencies TEXT[] DEFAULT '{}',
    mastery_scores JSONB DEFAULT '{}',
    last_assessment_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tabla para resultados de evaluaciones
CREATE TABLE IF NOT EXISTS assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    competency_id VARCHAR(100) NOT NULL,
    skill_type VARCHAR(50) NOT NULL, -- speaking, listening, reading, writing, vocabulary, grammar
    score FLOAT NOT NULL,
    max_score FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    feedback TEXT,
    areas_for_improvement TEXT[],
    strengths TEXT[],
    session_id UUID REFERENCES sessions(id)
);

-- Tabla para metas de aprendizaje personalizadas
CREATE TABLE IF NOT EXISTS learning_goals (
    id VARCHAR(100) PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    target_competencies TEXT[] NOT NULL,
    target_completion_date TIMESTAMP,
    created_date TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    progress_percentage FLOAT DEFAULT 0.0
);

-- Tabla para actividades de práctica completadas
CREATE TABLE IF NOT EXISTS practice_activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    competency_id VARCHAR(100) NOT NULL,
    activity_type VARCHAR(50) NOT NULL, -- vocabulary_drill, grammar_exercise, conversation_practice, etc.
    activity_data JSONB, -- datos específicos de la actividad
    score FLOAT,
    completed_at TIMESTAMP DEFAULT NOW(),
    time_spent_seconds INTEGER
);

-- Tabla para seguimiento del vocabulario aprendido
CREATE TABLE IF NOT EXISTS vocabulary_progress (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    word VARCHAR(100) NOT NULL,
    definition TEXT,
    competency_id VARCHAR(100),
    first_seen TIMESTAMP DEFAULT NOW(),
    times_practiced INTEGER DEFAULT 1,
    times_correct INTEGER DEFAULT 0,
    mastery_level FLOAT DEFAULT 0.0, -- 0-1, nivel de dominio
    last_practiced TIMESTAMP DEFAULT NOW(),
    next_review TIMESTAMP, -- para sistema de repetición espaciada
    UNIQUE(user_id, word)
);

-- Tabla para errores de gramática identificados
CREATE TABLE IF NOT EXISTS grammar_errors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id),
    error_type VARCHAR(100) NOT NULL, -- subject_verb_agreement, verb_tense, article_usage, etc.
    original_text TEXT NOT NULL,
    corrected_text TEXT,
    competency_id VARCHAR(100),
    context TEXT, -- contexto donde ocurrió el error
    detected_at TIMESTAMP DEFAULT NOW(),
    is_corrected BOOLEAN DEFAULT false,
    correction_explanation TEXT
);

-- Funciones para actualizar timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers para actualizar updated_at automáticamente
DROP TRIGGER IF EXISTS update_user_progress_updated_at ON user_progress;
CREATE TRIGGER update_user_progress_updated_at 
    BEFORE UPDATE ON user_progress 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Índices para optimización
CREATE INDEX IF NOT EXISTS idx_assessments_user_id ON assessments(user_id);
CREATE INDEX IF NOT EXISTS idx_assessments_competency ON assessments(competency_id);
CREATE INDEX IF NOT EXISTS idx_assessments_timestamp ON assessments(timestamp);

CREATE INDEX IF NOT EXISTS idx_learning_goals_user_id ON learning_goals(user_id);
CREATE INDEX IF NOT EXISTS idx_learning_goals_active ON learning_goals(is_active);

CREATE INDEX IF NOT EXISTS idx_practice_activities_user_id ON practice_activities(user_id);
CREATE INDEX IF NOT EXISTS idx_practice_activities_competency ON practice_activities(competency_id);
CREATE INDEX IF NOT EXISTS idx_practice_activities_completed ON practice_activities(completed_at);

CREATE INDEX IF NOT EXISTS idx_vocabulary_user_id ON vocabulary_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_vocabulary_word ON vocabulary_progress(word);
CREATE INDEX IF NOT EXISTS idx_vocabulary_mastery ON vocabulary_progress(mastery_level);
CREATE INDEX IF NOT EXISTS idx_vocabulary_next_review ON vocabulary_progress(next_review);

CREATE INDEX IF NOT EXISTS idx_grammar_errors_user_id ON grammar_errors(user_id);
CREATE INDEX IF NOT EXISTS idx_grammar_errors_type ON grammar_errors(error_type);
CREATE INDEX IF NOT EXISTS idx_grammar_errors_detected ON grammar_errors(detected_at);

CREATE INDEX IF NOT EXISTS idx_user_progress_level ON user_progress(current_level);
CREATE INDEX IF NOT EXISTS idx_user_progress_updated ON user_progress(updated_at);

-- Vista para estadísticas rápidas de progreso
CREATE OR REPLACE VIEW user_progress_summary AS
SELECT 
    u.id as user_id,
    u.phone_number,
    u.name,
    up.current_level,
    up.level_start_date,
    array_length(up.completed_competencies, 1) as completed_count,
    array_length(up.in_progress_competencies, 1) as in_progress_count,
    up.last_assessment_date,
    EXTRACT(DAYS FROM (NOW() - up.level_start_date)) as days_in_level,
    COUNT(a.id) as total_assessments,
    AVG(a.score / a.max_score) as avg_score_percentage
FROM users u
LEFT JOIN user_progress up ON u.id = up.user_id
LEFT JOIN assessments a ON u.id = a.user_id AND a.timestamp >= up.level_start_date
GROUP BY u.id, u.phone_number, u.name, up.current_level, up.level_start_date, 
         up.completed_competencies, up.in_progress_competencies, up.last_assessment_date;

-- Vista para análisis de vocabulario
CREATE OR REPLACE VIEW vocabulary_stats AS
SELECT 
    user_id,
    COUNT(*) as total_words,
    COUNT(*) FILTER (WHERE mastery_level >= 0.8) as mastered_words,
    COUNT(*) FILTER (WHERE mastery_level < 0.5) as struggling_words,
    AVG(mastery_level) as avg_mastery,
    COUNT(*) FILTER (WHERE last_practiced >= NOW() - INTERVAL '7 days') as practiced_this_week
FROM vocabulary_progress
GROUP BY user_id;

-- Vista para análisis de errores gramaticales
CREATE OR REPLACE VIEW grammar_error_analysis AS
SELECT 
    user_id,
    error_type,
    COUNT(*) as error_count,
    COUNT(*) FILTER (WHERE is_corrected = true) as corrected_count,
    AVG(CASE WHEN is_corrected THEN 1.0 ELSE 0.0 END) as correction_rate,
    MAX(detected_at) as last_error_date
FROM grammar_errors
GROUP BY user_id, error_type; 