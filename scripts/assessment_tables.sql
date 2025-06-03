-- Tablas para el sistema de evaluación automática y detección de nivel

-- Tabla para almacenar evaluaciones de conversaciones
CREATE TABLE IF NOT EXISTS conversation_assessments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id UUID NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    timestamp TIMESTAMP DEFAULT NOW(),
    user_message TEXT NOT NULL,
    
    -- Resultados de evaluación
    overall_level VARCHAR(2) NOT NULL, -- A1, A2, B1, B2, C1, C2
    confidence_score FLOAT NOT NULL, -- 0-1
    skills_scores JSONB, -- puntuaciones por habilidad
    
    -- Feedback generado
    strengths TEXT[],
    areas_for_improvement TEXT[],
    competency_evidence JSONB, -- evidencia de competencias específicas
    
    -- Análisis detallados
    vocab_level VARCHAR(2),
    grammar_level VARCHAR(2),
    vocab_complexity FLOAT, -- 0-1
    grammar_accuracy FLOAT, -- 0-1
    fluency_score FLOAT, -- 0-1
    
    -- Metadatos
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tabla para detecciones de nivel agregadas
CREATE TABLE IF NOT EXISTS level_detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    detected_level VARCHAR(2) NOT NULL,
    confidence FLOAT NOT NULL,
    evidence JSONB, -- evidencia por habilidad
    recommendation TEXT,
    should_advance BOOLEAN DEFAULT false,
    should_review BOOLEAN DEFAULT false,
    assessment_count INTEGER, -- número de evaluaciones consideradas
    detection_date TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true -- la detección más reciente
);

-- Tabla para progresión de habilidades
CREATE TABLE IF NOT EXISTS skill_progressions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    skill VARCHAR(50) NOT NULL, -- vocabulary, grammar, fluency, etc.
    current_score FLOAT NOT NULL, -- 0-1
    trend VARCHAR(20) NOT NULL, -- improving, stable, declining
    level_estimate VARCHAR(2) NOT NULL,
    next_milestone TEXT,
    scores_history JSONB, -- historial de puntuaciones
    last_updated TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, skill)
);

-- Tabla para almacenar errores detectados automáticamente
CREATE TABLE IF NOT EXISTS detected_errors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    assessment_id UUID REFERENCES conversation_assessments(id) ON DELETE CASCADE,
    error_type VARCHAR(100) NOT NULL,
    original_text TEXT NOT NULL,
    corrected_text TEXT,
    explanation TEXT,
    error_position JSONB, -- posición del error en el texto
    severity VARCHAR(20), -- minor, moderate, major, critical
    detected_at TIMESTAMP DEFAULT NOW()
);

-- Tabla para configuración de evaluación por usuario
CREATE TABLE IF NOT EXISTS assessment_configs (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    min_words_for_assessment INTEGER DEFAULT 5,
    assessment_frequency INTEGER DEFAULT 3,
    confidence_threshold FLOAT DEFAULT 0.7,
    level_change_threshold FLOAT DEFAULT 0.8,
    grammar_weight FLOAT DEFAULT 0.3,
    vocabulary_weight FLOAT DEFAULT 0.3,
    fluency_weight FLOAT DEFAULT 0.2,
    accuracy_weight FLOAT DEFAULT 0.2,
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Índices para optimización
CREATE INDEX IF NOT EXISTS idx_conversation_assessments_user_id ON conversation_assessments(user_id);
CREATE INDEX IF NOT EXISTS idx_conversation_assessments_timestamp ON conversation_assessments(timestamp);
CREATE INDEX IF NOT EXISTS idx_conversation_assessments_level ON conversation_assessments(overall_level);
CREATE INDEX IF NOT EXISTS idx_conversation_assessments_session ON conversation_assessments(session_id);

CREATE INDEX IF NOT EXISTS idx_level_detections_user_id ON level_detections(user_id);
CREATE INDEX IF NOT EXISTS idx_level_detections_active ON level_detections(is_active);
CREATE INDEX IF NOT EXISTS idx_level_detections_date ON level_detections(detection_date);

CREATE INDEX IF NOT EXISTS idx_skill_progressions_user_id ON skill_progressions(user_id);
CREATE INDEX IF NOT EXISTS idx_skill_progressions_skill ON skill_progressions(skill);
CREATE INDEX IF NOT EXISTS idx_skill_progressions_updated ON skill_progressions(last_updated);

CREATE INDEX IF NOT EXISTS idx_detected_errors_user_id ON detected_errors(user_id);
CREATE INDEX IF NOT EXISTS idx_detected_errors_type ON detected_errors(error_type);
CREATE INDEX IF NOT EXISTS idx_detected_errors_detected ON detected_errors(detected_at);

-- Trigger para marcar detecciones anteriores como inactivas
CREATE OR REPLACE FUNCTION update_level_detection_status()
RETURNS TRIGGER AS $$
BEGIN
    -- Marcar detecciones anteriores como inactivas
    UPDATE level_detections 
    SET is_active = false 
    WHERE user_id = NEW.user_id AND id != NEW.id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_level_detection_status ON level_detections;
CREATE TRIGGER trigger_update_level_detection_status
    AFTER INSERT ON level_detections
    FOR EACH ROW
    EXECUTE FUNCTION update_level_detection_status();

-- Trigger para actualizar timestamp en skill_progressions
CREATE OR REPLACE FUNCTION update_skill_progression_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.last_updated = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trigger_update_skill_progression_timestamp ON skill_progressions;
CREATE TRIGGER trigger_update_skill_progression_timestamp
    BEFORE UPDATE ON skill_progressions
    FOR EACH ROW
    EXECUTE FUNCTION update_skill_progression_timestamp();

-- Vistas para análisis y reportes
CREATE OR REPLACE VIEW assessment_summary AS
SELECT 
    u.id as user_id,
    u.phone_number,
    u.name,
    u.current_level as profile_level,
    COUNT(ca.id) as total_assessments,
    AVG(ca.confidence_score) as avg_confidence,
    MODE() WITHIN GROUP (ORDER BY ca.overall_level) as most_common_level,
    MAX(ca.timestamp) as last_assessment,
    AVG(ca.vocab_complexity) as avg_vocab_complexity,
    AVG(ca.grammar_accuracy) as avg_grammar_accuracy,
    AVG(ca.fluency_score) as avg_fluency_score
FROM users u
LEFT JOIN conversation_assessments ca ON u.id = ca.user_id
WHERE ca.timestamp >= NOW() - INTERVAL '30 days' -- últimos 30 días
GROUP BY u.id, u.phone_number, u.name, u.current_level;

CREATE OR REPLACE VIEW skill_overview AS
SELECT 
    sp.user_id,
    u.phone_number,
    sp.skill,
    sp.current_score,
    sp.trend,
    sp.level_estimate,
    sp.next_milestone,
    sp.last_updated
FROM skill_progressions sp
JOIN users u ON sp.user_id = u.id
ORDER BY sp.user_id, sp.skill;

CREATE OR REPLACE VIEW error_patterns AS
SELECT 
    user_id,
    error_type,
    COUNT(*) as error_count,
    AVG(CASE WHEN severity = 'critical' THEN 4 
             WHEN severity = 'major' THEN 3
             WHEN severity = 'moderate' THEN 2
             WHEN severity = 'minor' THEN 1
             ELSE 0 END) as avg_severity_score,
    MAX(detected_at) as last_occurrence,
    array_agg(DISTINCT explanation) as explanations
FROM detected_errors
WHERE detected_at >= NOW() - INTERVAL '30 days'
GROUP BY user_id, error_type
ORDER BY user_id, error_count DESC; 