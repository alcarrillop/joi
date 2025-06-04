-- Migration: Add user_word_stats table for vocabulary frequency tracking
-- This replaces the JSON array approach with a proper relational table

-- 1. Nueva tabla para tracking de palabras con frecuencias
CREATE TABLE IF NOT EXISTS user_word_stats (
    user_id      UUID        REFERENCES users(id) ON DELETE CASCADE,
    word         TEXT        NOT NULL,
    freq         INT         NOT NULL DEFAULT 1,
    last_used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, word)
);

-- 2. Índices para optimizar consultas
CREATE INDEX IF NOT EXISTS idx_word_stats_user_freq ON user_word_stats (user_id, freq DESC);
CREATE INDEX IF NOT EXISTS idx_word_stats_last_used ON user_word_stats (user_id, last_used_at DESC);

-- 3. Función para incrementar frecuencia de palabras (upsert)
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

-- 4. Función para obtener conteo total de vocabulario
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

-- 5. Función para obtener las palabras más frecuentes
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

-- 6. Migrar datos existentes desde learning_stats.vocab_learned
-- Esto convierte el array TEXT[] actual a la nueva tabla
INSERT INTO user_word_stats (user_id, word, freq, created_at)
SELECT 
    ls.user_id, 
    lower(trim(w)) as word,
    1 as freq,
    COALESCE(ls.last_updated, NOW()) as created_at
FROM learning_stats ls,
     unnest(ls.vocab_learned) AS w
WHERE ls.vocab_learned IS NOT NULL 
  AND array_length(ls.vocab_learned, 1) > 0
ON CONFLICT (user_id, word) DO NOTHING;

-- 7. Comentarios para documentación
COMMENT ON TABLE user_word_stats IS 'Tabla relacional para tracking de vocabulario con frecuencias de uso';
COMMENT ON COLUMN user_word_stats.word IS 'Palabra en forma singular/base normalizada';
COMMENT ON COLUMN user_word_stats.freq IS 'Número de veces que el usuario ha usado esta palabra';
COMMENT ON COLUMN user_word_stats.last_used_at IS 'Última vez que el usuario usó esta palabra';

COMMENT ON FUNCTION inc_word_freq(UUID, TEXT) IS 'Incrementa la frecuencia de una palabra para un usuario';
COMMENT ON FUNCTION get_user_vocab_count(UUID) IS 'Obtiene el conteo total de vocabulario de un usuario';
COMMENT ON FUNCTION get_user_top_words(UUID, INT) IS 'Obtiene las palabras más frecuentes de un usuario'; 