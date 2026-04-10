"""
Voice Interface — plug-in point for local speech I/O.

NOT ACTIVE IN V1. This module defines the contract only.

When voice is enabled later:
  Speech-to-text: local Whisper (openai-whisper) or faster-whisper
    - Runs entirely on MacBook — no audio sent to any server
    - Model: whisper-base or whisper-small (good for trading vocabulary)
    - Trigger: push-to-talk hotkey or always-on VAD

  Text-to-speech: macOS built-in `say` command (zero latency, no install)
    - Fallback: Piper TTS (offline neural TTS, high quality)
    - Only used for alerts and proactive commentary, not full responses

  Integration pattern:
    - VoiceInterface.listen() → str  (same as user typing)
    - VoiceInterface.speak(text)     (same as CLI printing)
    - The AI layer is completely unaware of whether input came from keyboard or voice

To activate voice in a future version:
  1. pip install openai-whisper (or faster-whisper)
  2. Set VOICE_ENABLED=true in .env
  3. Instantiate WhisperVoiceInterface (subclass below) in main.py
  4. Pass it to the CLI dashboard as input/output adapter

The system works identically with or without voice.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class VoiceInterface(ABC):
    """
    Abstract voice I/O contract.
    The CLI and voice share this interface — the AI layer calls neither directly.
    """

    @abstractmethod
    async def listen(self) -> Optional[str]:
        """
        Record and transcribe user speech.
        Returns transcribed text, or None if nothing captured.
        """
        ...

    @abstractmethod
    async def speak(self, text: str) -> None:
        """
        Synthesize and play text-to-speech response.
        Non-blocking — fires and forgets.
        """
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """True if the voice hardware and model are ready."""
        ...


class NullVoiceInterface(VoiceInterface):
    """
    No-op voice interface. Used in v1 — all operations are silent.
    Replace with WhisperVoiceInterface when voice is activated.
    """

    async def listen(self) -> Optional[str]:
        return None

    async def speak(self, text: str) -> None:
        pass

    @property
    def is_available(self) -> bool:
        return False


# ──────────────────────────────────────────────
# Future: WhisperVoiceInterface
# ──────────────────────────────────────────────
#
# class WhisperVoiceInterface(VoiceInterface):
#     """
#     Local Whisper STT + macOS `say` TTS.
#     Activate by: pip install openai-whisper pyaudio
#
#     def __init__(self, model_size: str = "base"):
#         import whisper
#         self._model = whisper.load_model(model_size)
#         ...
#
#     async def listen(self) -> Optional[str]:
#         audio = self._record_audio(seconds=5)
#         result = self._model.transcribe(audio)
#         return result["text"].strip() or None
#
#     async def speak(self, text: str) -> None:
#         import subprocess
#         # macOS `say` — instant, offline, zero latency
#         subprocess.Popen(["say", "-v", "Samantha", text])
#     """
#
# ──────────────────────────────────────────────


# Default export
def get_voice_interface(enabled: bool = False) -> VoiceInterface:
    """
    Factory. Returns NullVoiceInterface until voice is activated.
    When activating: return WhisperVoiceInterface() here.
    """
    return NullVoiceInterface()
