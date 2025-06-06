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

-- 4. Tabla moderna de estadísticas de vocabulario (reemplaza learning_stats)
CREATE TABLE IF NOT EXISTS user_word_stats (
    user_id      UUID        REFERENCES users(id) ON DELETE CASCADE,
    word         TEXT        NOT NULL,
    freq         INT         NOT NULL DEFAULT 1,
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, word)
);

-- Índices para mejorar el rendimiento
CREATE INDEX IF NOT EXISTS idx_users_phone_number ON users(phone_number);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_word_stats_user_freq ON user_word_stats (user_id, freq DESC);
CREATE INDEX IF NOT EXISTS idx_word_stats_last_used ON user_word_stats (user_id, last_used_at DESC);

-- Funciones PostgreSQL para manejo de vocabulario

-- Función para incrementar frecuencia de palabras (upsert)
CREATE OR REPLACE FUNCTION inc_word_freq(p_user UUID, p_word TEXT)
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO user_word_stats (user_id, word, freq)
    VALUES (p_user, lower(trim(p_word)), 1)
    ON CONFLICT (user_id, word)
    DO UPDATE SET
        freq = user_word_stats.freq + 1,
        last_used_at = NOW();
END;
$$;

-- Función para obtener conteo total de vocabulario
CREATE OR REPLACE FUNCTION get_user_vocab_count(p_user UUID)
RETURNS INT LANGUAGE plpgsql AS $$
DECLARE
    word_count INT;
BEGIN
    SELECT COUNT(*) INTO word_count
    FROM user_word_stats
    WHERE user_id = p_user;
    
    RETURN COALESCE(word_count, 0);
END;
$$;

-- Función para obtener las palabras más frecuentes
CREATE OR REPLACE FUNCTION get_user_top_words(p_user UUID, p_limit INT DEFAULT 10)
RETURNS TABLE(word TEXT, frequency INT, last_used TIMESTAMPTZ) 
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT uws.word, uws.freq, uws.last_used_at
    FROM user_word_stats uws
    WHERE uws.user_id = p_user
    ORDER BY uws.freq DESC, uws.last_used_at DESC
    LIMIT p_limit;
END;
$$;

-- Comentarios para documentación
COMMENT ON TABLE users IS 'Usuarios registrados del sistema de aprendizaje';
COMMENT ON TABLE sessions IS 'Sesiones de conversación simplificadas para agrupar mensajes';
COMMENT ON TABLE messages IS 'Mensajes individuales dentro de sesiones';
COMMENT ON TABLE user_word_stats IS 'Tabla relacional para tracking de vocabulario con frecuencias de uso';

COMMENT ON COLUMN user_word_stats.word IS 'Palabra en forma singular/base normalizada';
COMMENT ON COLUMN user_word_stats.freq IS 'Número de veces que el usuario ha usado esta palabra';
COMMENT ON COLUMN user_word_stats.last_used_at IS 'Última vez que el usuario usó esta palabra';

COMMENT ON FUNCTION inc_word_freq(UUID, TEXT) IS 'Incrementa la frecuencia de una palabra para un usuario';
COMMENT ON FUNCTION get_user_vocab_count(UUID) IS 'Obtiene el conteo total de vocabulario de un usuario';
COMMENT ON FUNCTION get_user_top_words(UUID, INT) IS 'Obtiene las palabras más frecuentes de un usuario'; 