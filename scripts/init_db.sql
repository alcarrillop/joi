-- Schema inicial para Joi - AI Language Learning Assistant
-- Este script crea las tablas relacionales necesarias para gestionar usuarios,
-- sesiones, mensajes y estadísticas de aprendizaje
-- 
-- Sistema simplificado enfocado en memoria y aprendizaje básico

-- 1. Tabla de usuarios
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    phone_number TEXT UNIQUE NOT NULL,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 2. Tabla de sesiones (simplificada para alinearse con LangGraph)
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    started_at TIMESTAMPTZ DEFAULT now()
);

-- 3. Tabla de mensajes
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id),
    sender TEXT CHECK (sender IN ('user', 'agent')),
    message TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT now()
);

-- 4. Tabla de estadísticas de aprendizaje
CREATE TABLE IF NOT EXISTS learning_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    vocab_learned TEXT[],
    grammar_issues JSONB,
    last_updated TIMESTAMPTZ DEFAULT now()
);

-- Índices para mejorar el rendimiento
CREATE INDEX IF NOT EXISTS idx_users_phone_number ON users(phone_number);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_learning_stats_user_id ON learning_stats(user_id);

-- Comentarios para documentación
COMMENT ON TABLE users IS 'Usuarios registrados del sistema de aprendizaje';
COMMENT ON TABLE sessions IS 'Sesiones de conversación simplificadas para agrupar mensajes';
COMMENT ON TABLE messages IS 'Mensajes individuales dentro de sesiones';
COMMENT ON TABLE learning_stats IS 'Estadísticas y progreso de aprendizaje por usuario';

COMMENT ON COLUMN learning_stats.vocab_learned IS 'Array de vocabulario aprendido por el usuario';
COMMENT ON COLUMN learning_stats.grammar_issues IS 'Errores gramaticales frecuentes en formato JSON'; 