// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! Platform-agnostic Speech-to-Text (STT) abstractions and providers.
//!
//! Provides a unified [`SpeechToTextProvider`] trait for transcribing audio into
//! timestamped text segments and word-level timing tokens, an [`AudioPreprocessor`]
//! to extract/transcode and chunk audio to meet provider limits, a [`LanguageModelRouter`]
//! for mapping languages to provider models with auto-mode fallback, and concrete
//! implementations such as [`ElevenLabsProvider`].

pub mod elevenlabs;
pub mod preprocessor;
pub mod whisper;

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

pub use elevenlabs::ElevenLabsProvider;
pub use preprocessor::AudioPreprocessor;
pub use whisper::WhisperProvider;
use serde::{Deserialize, Serialize};

use crate::transcribe::Segment;

/// The type of token returned by word-level transcription models.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenType {
    /// A standard spoken word.
    Word,
    /// Whitespace between words.
    Spacing,
    /// Detected non-speech audio event (e.g. laughter, applause, music).
    AudioEvent,
}

impl Default for TokenType {
    fn default() -> Self {
        Self::Word
    }
}

/// A granular word-level token with exact timing and optional speaker attribution.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct WordSegment {
    /// The word text or event description (e.g. "(laughter)").
    pub text: String,
    /// Seconds from the start of the audio window.
    pub start: f64,
    /// Seconds from the start of the audio window.
    pub end: f64,
    /// Type of token (word, spacing, audio event).
    pub token_type: TokenType,
    /// Optional speaker identifier from diarization (e.g. "speaker_0").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_id: Option<String>,
}

/// The aggregated result of a speech-to-text transcription run.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionResult {
    /// The full transcribed text.
    pub text: String,
    /// Timeline subtitle segments (phrases/sentences).
    pub segments: Vec<Segment>,
    /// Granular word-level tokens for animations and karaoke.
    pub words: Vec<WordSegment>,
    /// The detected or specified language code (e.g. "en", "es").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detected_language: Option<String>,
    /// Confidence probability of the detected language (0.0 to 1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language_probability: Option<f64>,
}

/// Descriptor for a transcription model offered by a provider.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ModelDescriptor {
    /// Model identifier passed to the API (e.g. "scribe_v2", "base.en").
    pub id: String,
    /// Human-friendly display name.
    pub name: String,
    /// Description of the model's strengths or tradeoffs.
    pub description: String,
}

/// Request payload handed to a [`SpeechToTextProvider`].
#[derive(Clone, Debug)]
pub struct ProviderRequest {
    /// Encoded audio bytes (e.g. WAV, MP3, AAC).
    pub audio_bytes: Vec<u8>,
    /// Virtual filename for multipart uploads (e.g. "audio.wav", "audio.mp3").
    pub filename: String,
    /// MIME content type of `audio_bytes`.
    pub mime_type: String,
    /// Model identifier to transcribe with.
    pub model_id: String,
    /// Language code (e.g. "en", "es"), or `None` / "auto" for automatic detection.
    pub language: Option<String>,
    /// Whether to detect and label non-speech sounds (e.g. laughter, applause).
    pub tag_audio_events: bool,
    /// Whether to run speaker diarization.
    pub diarize: bool,
    /// Expected number of speakers if known.
    pub num_speakers: Option<u32>,
}

/// Platform-agnostic trait for Speech-to-Text providers.
pub trait SpeechToTextProvider: Send + Sync {
    /// Stable identifier of the provider (e.g. "elevenlabs", "whisper").
    fn id(&self) -> &'static str;

    /// Human-readable display name (e.g. "ElevenLabs Scribe V2", "Local Whisper").
    fn name(&self) -> &'static str;

    /// List of models supported by this provider.
    fn available_models(&self) -> Vec<ModelDescriptor>;

    /// The default model ID when none is specified.
    fn default_model(&self) -> &'static str;

    /// Maximum file payload size in bytes accepted by this provider before chunking.
    fn max_payload_bytes(&self) -> usize {
        25 * 1024 * 1024 // Default 25 MB
    }

    /// Run transcription against this provider.
    fn transcribe(
        &self,
        request: &ProviderRequest,
        cancel: &AtomicBool,
    ) -> Result<TranscriptionResult, String>;
}

/// Routes language codes to specific model IDs, falling back to a default model.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LanguageModelRouter {
    /// The default model used when language is omitted, "auto", or unmapped.
    pub default_model: String,
    /// Explicit mapping of language code (ISO 639-1 or ISO 639-3) to model ID.
    #[serde(default)]
    pub language_map: HashMap<String, String>,
}

impl LanguageModelRouter {
    /// Create a new router with a default model.
    pub fn new(default_model: impl Into<String>) -> Self {
        Self {
            default_model: default_model.into(),
            language_map: HashMap::new(),
        }
    }

    /// Add a language mapping.
    pub fn with_mapping(mut self, language: impl Into<String>, model_id: impl Into<String>) -> Self {
        self.language_map
            .insert(language.into().to_ascii_lowercase(), model_id.into());
        self
    }

    /// Resolve which model to use for the given language.
    ///
    /// If `language` is `None`, empty, or "auto" (case-insensitive), returns `default_model`.
    /// Otherwise, checks `language_map` (case-insensitive). If found, returns that model,
    /// else falls back to `default_model`.
    pub fn resolve(&self, language: Option<&str>) -> &str {
        let Some(raw) = language else {
            return &self.default_model;
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
            return &self.default_model;
        }

        let lower = trimmed.to_ascii_lowercase();
        if let Some(model) = self.language_map.get(&lower) {
            return model.as_str();
        }

        // Check if an ISO 639-1 prefix matches (e.g. "en-US" -> "en")
        if let Some((prefix, _)) = lower.split_once('-') {
            if let Some(model) = self.language_map.get(prefix) {
                return model.as_str();
            }
        }

        &self.default_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_model_router() {
        let mut router = LanguageModelRouter::new("scribe_v2");
        router
            .language_map
            .insert("en-med".to_string(), "scribe_v2_medical".to_string());
        router
            .language_map
            .insert("es".to_string(), "scribe_v2".to_string());

        assert_eq!(router.resolve(None), "scribe_v2");
        assert_eq!(router.resolve(Some("")), "scribe_v2");
        assert_eq!(router.resolve(Some("auto")), "scribe_v2");
        assert_eq!(router.resolve(Some("AUTO")), "scribe_v2");
        assert_eq!(router.resolve(Some("en-med")), "scribe_v2_medical");
        assert_eq!(router.resolve(Some("EN-MED")), "scribe_v2_medical");
        assert_eq!(router.resolve(Some("es")), "scribe_v2");
        assert_eq!(router.resolve(Some("fr")), "scribe_v2");
    }
}
