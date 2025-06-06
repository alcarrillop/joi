-- Migration: Add message_type column to messages table
-- This adds support for tracking different types of messages

-- Add message_type column with valid types
ALTER TABLE messages 
ADD COLUMN IF NOT EXISTS message_type TEXT 
CHECK (message_type IN ('text', 'audio', 'image')) 
DEFAULT 'text';

-- Create index for message type queries
CREATE INDEX IF NOT EXISTS idx_messages_type ON messages (message_type);

-- Add comment for documentation
COMMENT ON COLUMN messages.message_type IS 'Type of message: text, audio, or image. Users can send all types, agents send text/audio only';

-- Update existing messages to have 'text' type (they were all text previously)
UPDATE messages 
SET message_type = 'text' 
WHERE message_type IS NULL; 