import os
from typing import Optional

from elevenlabs import ElevenLabs, Voice, VoiceSettings

from agent.core.exceptions import TextToSpeechError
from agent.settings import get_settings


class TextToSpeech:
    """A class to handle text-to-speech conversion using ElevenLabs."""

    # Required environment variables
    REQUIRED_ENV_VARS = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]

    def __init__(self):
        """Initialize the TextToSpeech class and validate environment variables."""
        self._validate_env_vars()
        settings = get_settings()
        self.voice_id = settings.ELEVENLABS_VOICE_ID
        self.model_id = settings.TTS_MODEL_NAME
        self._client: Optional[ElevenLabs] = None

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def client(self) -> ElevenLabs:
        """Get or create ElevenLabs client instance using singleton pattern."""
        if self._client is None:
            settings = get_settings()
            self._client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        return self._client

    async def synthesize(self, text: str) -> bytes:
        """Convert text to speech using ElevenLabs.

        Args:
            text: Text to convert to speech

        Returns:
            bytes: Audio data

        Raises:
            ValueError: If the input text is empty or too long
            TextToSpeechError: If the text-to-speech conversion fails
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        if len(text) > 5000:  # ElevenLabs typical limit
            raise ValueError("Input text exceeds maximum length of 5000 characters")

        try:
            audio_generator = self.client.generate(
                text=text,
                voice=Voice(
                    voice_id=self.voice_id,
                    settings=VoiceSettings(stability=0.5, similarity_boost=0.5),
                ),
                model=self.model_id,
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)
            if not audio_bytes:
                raise TextToSpeechError("Generated audio is empty")

            return audio_bytes

        except Exception as e:
            raise TextToSpeechError(f"Text-to-speech conversion failed: {str(e)}") from e
