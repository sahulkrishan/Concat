// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! Local Whisper implementation of [`SpeechToTextProvider`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use concat_host::AppDirs;

use super::{ModelDescriptor, ProviderRequest, SpeechToTextProvider, TranscriptionResult};
use crate::transcribe::{KNOWN_MODELS, TranscribeRequest, Transcriber};

/// Speech-to-text provider running Whisper locally in-process.
#[derive(Clone)]
pub struct WhisperProvider {
    /// In-process Transcriber instance.
    pub transcriber: Arc<Transcriber>,
    /// Application directory paths for models.
    pub dirs: AppDirs,
}

impl WhisperProvider {
    /// Create a new local Whisper provider.
    pub fn new(transcriber: Arc<Transcriber>, dirs: AppDirs) -> Self {
        Self { transcriber, dirs }
    }
}

impl SpeechToTextProvider for WhisperProvider {
    fn id(&self) -> &'static str {
        "whisper"
    }

    fn name(&self) -> &'static str {
        "Local Whisper"
    }

    fn default_model(&self) -> &'static str {
        "base"
    }

    fn available_models(&self) -> Vec<ModelDescriptor> {
        KNOWN_MODELS
            .iter()
            .map(|model| ModelDescriptor {
                id: model.id.to_owned(),
                name: model.label.to_owned(),
                description: model.blurb.to_owned(),
            })
            .collect()
    }

    fn max_payload_bytes(&self) -> usize {
        // Local Whisper runs in memory without HTTP payload limits
        usize::MAX
    }

    fn transcribe(
        &self,
        request: &ProviderRequest,
        cancel: &AtomicBool,
    ) -> Result<TranscriptionResult, String> {
        if cancel.load(Ordering::Relaxed) {
            return Err("transcription cancelled".to_owned());
        }

        // Write the audio bytes to a temporary WAV file for the audio decoder
        let temp_dir = std::env::temp_dir().join("concat-whisper");
        std::fs::create_dir_all(&temp_dir)
            .map_err(|e| format!("failed to create temp dir: {e}"))?;

        let temp_file = temp_dir.join(format!(
            "input_{}_{:x}.wav",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));

        std::fs::write(&temp_file, &request.audio_bytes)
            .map_err(|e| format!("failed to write temp audio file: {e}"))?;

        let transcribe_req = TranscribeRequest {
            path: temp_file.to_string_lossy().into_owned(),
            audio_stream: None,
            source_start: 0.0,
            window: 0.0, // Handled automatically by length of audio file
            model_id: if request.model_id.is_empty() {
                self.default_model().to_owned()
            } else {
                request.model_id.clone()
            },
            language: request.language.clone(),
        };

        // Run transcription
        let segments_res = self
            .transcriber
            .transcribe(&self.dirs, &transcribe_req, |_| {});

        let _ = std::fs::remove_file(&temp_file);

        let segments = segments_res?;
        let full_text = segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        let words: Vec<_> = segments
            .iter()
            .flat_map(|s| s.words.iter().cloned())
            .collect();

        Ok(TranscriptionResult {
            text: full_text,
            segments,
            words,
            detected_language: request.language.clone(),
            language_probability: None,
        })
    }
}
