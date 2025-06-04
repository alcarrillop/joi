-- Migration: Remove obsolete learning_stats table
-- This table has been replaced by the user_word_stats relational table

-- 1. First verify that user_word_stats table exists and has data
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'user_word_stats') THEN
        RAISE EXCEPTION 'user_word_stats table does not exist. Migration cannot proceed safely.';
    END IF;
    
    -- Check if we have vocabulary data in the new table
    IF (SELECT COUNT(*) FROM user_word_stats) = 0 THEN
        RAISE NOTICE 'Warning: user_word_stats table is empty. Proceeding with caution.';
    ELSE
        RAISE NOTICE 'user_word_stats table has % entries. Safe to proceed.', (SELECT COUNT(*) FROM user_word_stats);
    END IF;
END $$;

-- 2. Drop the obsolete learning_stats table
DROP TABLE IF EXISTS learning_stats CASCADE;

-- 3. Add comments for documentation
COMMENT ON TABLE user_word_stats IS 'Relational vocabulary tracking table (replaced learning_stats.vocab_learned array)';

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✅ Successfully removed obsolete learning_stats table';
    RAISE NOTICE '📊 Vocabulary tracking now uses user_word_stats table exclusively';
END $$; 