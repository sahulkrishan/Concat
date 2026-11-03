// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! ElevenLabs Scribe V2 speech-to-text API provider.
//!
//! Calls `POST /v1/speech-to-text` on ElevenLabs with multipart audio payloads,
//! handling `scribe_v2` and `scribe_v2_medical` models, speaker diarization,
//! audio event tagging, and word-level timestamps.

use std::sync::atomic::{AtomicBool, Ordering};

use serde::Deserialize;

use super::{
    ModelDescriptor, ProviderRequest, SpeechToTextProvider, TokenType, TranscriptionResult,
    WordSegment,
};
use crate::transcribe::Segment;

/// Default endpoint for ElevenLabs API.
pub const ELEVENLABS_DEFAULT_BASE_URL: &str = "https://api.elevenlabs.io";

/// ElevenLabs Scribe V2 provider implementation.
#[derive(Clone, Debug)]
pub struct ElevenLabsProvider {
    /// ElevenLabs API key.
    pub api_key: String,
    /// Base API URL (defaults to `https://api.elevenlabs.io`).
    pub base_url: String,
}

impl ElevenLabsProvider {
    /// Create a new ElevenLabs provider with the given API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: ELEVENLABS_DEFAULT_BASE_URL.to_owned(),
        }
    }

    /// Set a custom base URL (e.g. for proxies or testing).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

impl SpeechToTextProvider for ElevenLabsProvider {
    fn id(&self) -> &'static str {
        "elevenlabs"
    }

    fn name(&self) -> &'static str {
        "ElevenLabs Scribe V2"
    }

    fn default_model(&self) -> &'static str {
        "scribe_v2"
    }

    fn available_models(&self) -> Vec<ModelDescriptor> {
        vec![
            ModelDescriptor {
                id: "scribe_v2".to_owned(),
                name: "Scribe V2".to_owned(),
                description: "State-of-the-art accuracy across 90+ languages with word timestamps"
                    .to_owned(),
            },
            ModelDescriptor {
                id: "scribe_v2_medical".to_owned(),
                name: "Scribe V2 Medical".to_owned(),
                description: "Optimized for clinical, pharmaceutical, and medical terminology"
                    .to_owned(),
            },
        ]
    }

    fn max_payload_bytes(&self) -> usize {
        // ElevenLabs accepts up to 5 GB, but we default chunking threshold to 100 MB
        // to prevent gigantic memory allocations and timeouts.
        100 * 1024 * 1024
    }

    fn transcribe(
        &self,
        request: &ProviderRequest,
        cancel: &AtomicBool,
    ) -> Result<TranscriptionResult, String> {
        if cancel.load(Ordering::Relaxed) {
            return Err("transcription cancelled".to_owned());
        }

        let key = self.api_key.trim();
        if key.is_empty() {
            return Err("ElevenLabs API key is missing. Please configure it in Settings > Transcriber".to_owned());
        }

        let url = format!("{}/v1/speech-to-text", self.base_url.trim_end_matches('/'));
        let boundary = format!("----ConcatSTTBoundary{:x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        let body = build_multipart_body(request, &boundary)?;

        let response = ureq::post(&url)
            .set("xi-api-key", key)
            .set("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
            .timeout(std::time::Duration::from_secs(300))
            .send_bytes(&body);

        if cancel.load(Ordering::Relaxed) {
            return Err("transcription cancelled".to_owned());
        }

        match response {
            Ok(res) => {
                let status = res.status();
                let body_text = res
                    .into_string()
                    .map_err(|e| format!("failed to read response body: {e}"))?;

                if status != 200 {
                    return Err(format!("ElevenLabs API returned HTTP {status}: {body_text}"));
                }

                parse_elevenlabs_response(&body_text)
            }
            Err(ureq::Error::Status(status, res)) => {
                let err_text = res.into_string().unwrap_or_default();
                let message = extract_error_message(&err_text).unwrap_or(err_text);
                Err(format!("ElevenLabs API error (HTTP {status}): {message}"))
            }
            Err(ureq::Error::Transport(transport)) => {
                Err(format!("Network error connecting to ElevenLabs: {transport}"))
            }
        }
    }
}

/// Builds a `multipart/form-data` payload in memory.
pub fn build_multipart_body(
    request: &ProviderRequest,
    boundary: &str,
) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();

    // Helper to append a text field
    let append_field = |body: &mut Vec<u8>, name: &str, value: &str| {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    };

    // model_id
    let model = if request.model_id.trim().is_empty() {
        "scribe_v2"
    } else {
        request.model_id.as_str()
    };
    append_field(&mut body, "model_id", model);

    // language_code
    if let Some(ref lang) = request.language {
        let trimmed = lang.trim();
        if !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("auto") {
            append_field(&mut body, "language_code", trimmed);
        }
    }

    // tag_audio_events
    append_field(
        &mut body,
        "tag_audio_events",
        if request.tag_audio_events { "true" } else { "false" },
    );

    // diarize
    append_field(
        &mut body,
        "diarize",
        if request.diarize { "true" } else { "false" },
    );

    // timestamps_granularity
    append_field(&mut body, "timestamps_granularity", "word");

    // num_speakers (optional)
    if let Some(num) = request.num_speakers {
        append_field(&mut body, "num_speakers", &num.to_string());
    }

    // file
    let filename = if request.filename.is_empty() {
        "audio.wav"
    } else {
        request.filename.as_str()
    };
    let mime = if request.mime_type.is_empty() {
        "audio/wav"
    } else {
        request.mime_type.as_str()
    };

    body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body.extend_from_slice(
        format!(
            "Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(&request.audio_bytes);
    body.extend_from_slice(b"\r\n");

    // Closing boundary
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    Ok(body)
}

/// JSON deserialization structure for raw ElevenLabs Scribe V2 response.
#[derive(Deserialize, Debug)]
struct RawElevenLabsResponse {
    text: String,
    #[serde(default)]
    language_code: Option<String>,
    #[serde(default)]
    language_probability: Option<f64>,
    #[serde(default)]
    words: Vec<RawElevenLabsWord>,
}

#[derive(Deserialize, Debug)]
struct RawElevenLabsWord {
    text: String,
    start: f64,
    end: f64,
    #[serde(rename = "type", default)]
    token_type: Option<String>,
    #[serde(default)]
    speaker_id: Option<String>,
}

/// Parse ElevenLabs JSON string into [`TranscriptionResult`], grouping words into subtitle [`Segment`]s.
pub fn parse_elevenlabs_response(json_str: &str) -> Result<TranscriptionResult, String> {
    let raw: RawElevenLabsResponse = serde_json::from_str(json_str)
        .map_err(|e| format!("failed to parse ElevenLabs response JSON: {e}"))?;

    let mut words = Vec::with_capacity(raw.words.len());
    for w in raw.words {
        let token_type = match w.token_type.as_deref() {
            Some("spacing") => TokenType::Spacing,
            Some("audio_event") => TokenType::AudioEvent,
            _ => TokenType::Word,
        };

        words.push(WordSegment {
            text: w.text,
            start: w.start,
            end: w.end,
            token_type,
            speaker_id: w.speaker_id,
        });
    }

    let segments = group_words_into_segments(&words);

    Ok(TranscriptionResult {
        text: raw.text,
        segments,
        words,
        detected_language: raw.language_code,
        language_probability: raw.language_probability,
    })
}

/// Extract human-readable error detail from ElevenLabs error JSON if possible.
fn extract_error_message(json_str: &str) -> Option<String> {
    #[derive(Deserialize)]
    struct ErrorWrapper {
        detail: Option<serde_json::Value>,
    }

    let parsed: ErrorWrapper = serde_json::from_str(json_str).ok()?;
    match parsed.detail? {
        serde_json::Value::String(s) => Some(s),
        serde_json::Value::Object(obj) => {
            if let Some(serde_json::Value::String(msg)) = obj.get("message") {
                Some(msg.clone())
            } else {
                None
            }
        }
        serde_json::Value::Array(arr) => {
            let messages: Vec<String> = arr
                .iter()
                .filter_map(|v| v.get("msg").and_then(|m| m.as_str()).map(str::to_owned))
                .collect();
            if messages.is_empty() {
                None
            } else {
                Some(messages.join(", "))
            }
        }
        _ => None,
    }
}

/// Group a sequence of word tokens into readable subtitle segments (phrases).
///
/// Breaks segments on:
/// - Sentence boundary punctuation (`.`, `?`, `!`)
/// - Audio gaps / pauses greater than 0.8 seconds
/// - Segment duration exceeding ~5.0 seconds or ~10 words
pub fn group_words_into_segments(words: &[WordSegment]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current_words = Vec::new();
    let mut seg_start: Option<f64> = None;
    let mut last_end = 0.0;

    for word in words {
        // Skip pure spacing tokens for boundary timing, but keep word text
        if word.token_type == TokenType::Spacing {
            continue;
        }

        let is_pause = seg_start.is_some() && (word.start - last_end > 0.8);
        let exceeds_duration = seg_start.is_some_and(|start| word.end - start > 5.0);
        let exceeds_word_count = current_words.len() >= 10;

        if !current_words.is_empty() && (is_pause || exceeds_duration || exceeds_word_count) {
            if let Some(start) = seg_start {
                let text = build_segment_text(&current_words);
                if !text.is_empty() {
                    segments.push(Segment {
                        start,
                        end: last_end,
                        text,
                        words: current_words.clone(),
                    });
                }
            }
            current_words.clear();
            seg_start = None;
        }

        if seg_start.is_none() {
            seg_start = Some(word.start);
        }
        last_end = word.end;
        current_words.push(word.clone());

        // Check if word ends with sentence punctuation
        let ends_sentence = word.text.ends_with('.')
            || word.text.ends_with('?')
            || word.text.ends_with('!')
            || word.text.ends_with(';')
            || word.text.ends_with('。')
            || word.text.ends_with('？')
            || word.text.ends_with('！');

        if ends_sentence {
            if let Some(start) = seg_start {
                let text = build_segment_text(&current_words);
                if !text.is_empty() {
                    segments.push(Segment {
                        start,
                        end: last_end,
                        text,
                        words: current_words.clone(),
                    });
                }
            }
            current_words.clear();
            seg_start = None;
        }
    }

    // Flush any remaining words
    if let Some(start) = seg_start {
        let text = build_segment_text(&current_words);
        if !text.is_empty() {
            segments.push(Segment {
                start,
                end: last_end,
                text,
                words: current_words,
            });
        }
    }

    segments
}

fn build_segment_text(words: &[WordSegment]) -> String {
    let mut text = String::new();
    for (i, w) in words.iter().enumerate() {
        if i > 0 && !w.text.starts_with(['.', ',', '!', '?', ';', ':', '\'', '’']) {
            text.push(' ');
        }
        text.push_str(&w.text);
    }
    text.trim().to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_elevenlabs_response() {
        let sample_json = r#"{
            "language_code": "en",
            "language_probability": 0.99,
            "text": "Hello world (laughter)",
            "words": [
                {
                    "text": "Hello",
                    "start": 0.12,
                    "end": 0.45,
                    "type": "word",
                    "speaker_id": "speaker_0"
                },
                {
                    "text": " ",
                    "start": 0.45,
                    "end": 0.50,
                    "type": "spacing",
                    "speaker_id": "speaker_0"
                },
                {
                    "text": "world.",
                    "start": 0.50,
                    "end": 0.85,
                    "type": "word",
                    "speaker_id": "speaker_0"
                },
                {
                    "text": "(laughter)",
                    "start": 1.20,
                    "end": 1.80,
                    "type": "audio_event",
                    "speaker_id": "speaker_1"
                }
            ]
        }"#;

        let res = parse_elevenlabs_response(sample_json).expect("parses cleanly");
        assert_eq!(res.detected_language.as_deref(), Some("en"));
        assert_eq!(res.words.len(), 4);
        assert_eq!(res.words[0].text, "Hello");
        assert_eq!(res.words[0].token_type, TokenType::Word);
        assert_eq!(res.words[0].speaker_id.as_deref(), Some("speaker_0"));
        assert_eq!(res.words[3].token_type, TokenType::AudioEvent);

        // Verify subtitle segments
        assert_eq!(res.segments.len(), 2);
        assert_eq!(res.segments[0].text, "Hello world.");
        assert_eq!(res.segments[0].start, 0.12);
        assert_eq!(res.segments[0].end, 0.85);
        assert_eq!(res.segments[0].words.len(), 2);

        assert_eq!(res.segments[1].text, "(laughter)");
        assert_eq!(res.segments[1].start, 1.20);
        assert_eq!(res.segments[1].end, 1.80);
    }

    #[test]
    fn test_extract_error_message() {
        let err1 = r#"{"detail": {"message": "Invalid API key", "code": 401}}"#;
        assert_eq!(
            extract_error_message(err1).as_deref(),
            Some("Invalid API key")
        );

        let err2 = r#"{"detail": "File size exceeds limit"}"#;
        assert_eq!(
            extract_error_message(err2).as_deref(),
            Some("File size exceeds limit")
        );

        let err3 = r#"{"detail": [{"loc": ["file"], "msg": "field required"}]}"#;
        assert_eq!(extract_error_message(err3).as_deref(), Some("field required"));
    }

    #[test]
    fn test_build_multipart_body() {
        let req = ProviderRequest {
            audio_bytes: vec![1, 2, 3, 4],
            filename: "test.wav".to_owned(),
            mime_type: "audio/wav".to_owned(),
            model_id: "scribe_v2".to_owned(),
            language: Some("en".to_owned()),
            tag_audio_events: true,
            diarize: true,
            num_speakers: Some(2),
        };

        let body = build_multipart_body(&req, "BOUNDARY123").expect("builds body");
        let body_str = String::from_utf8_lossy(&body);

        assert!(body_str.contains("--BOUNDARY123"));
        assert!(body_str.contains("name=\"model_id\"\r\n\r\nscribe_v2"));
        assert!(body_str.contains("name=\"language_code\"\r\n\r\nen"));
        assert!(body_str.contains("name=\"tag_audio_events\"\r\n\r\ntrue"));
        assert!(body_str.contains("name=\"diarize\"\r\n\r\ntrue"));
        assert!(body_str.contains("name=\"num_speakers\"\r\n\r\n2"));
        assert!(body_str.contains("name=\"file\"; filename=\"test.wav\""));
    }

    #[test]
    fn test_elevenlabs_provider_roundtrip_mock() {
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").expect("binds random port");
        let port = listener.local_addr().unwrap().port();

        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap();
            let req_str = String::from_utf8_lossy(&buf[..n]);
            assert!(req_str.contains("POST /v1/speech-to-text"));
            assert!(req_str.contains("xi-api-key: test-key-123"));

            let mock_response = r#"{
                "language_code": "en",
                "text": "Hello world from ElevenLabs",
                "words": [
                    {"text": "Hello", "start": 0.0, "end": 0.5, "type": "word"},
                    {"text": "world", "start": 0.6, "end": 1.0, "type": "word"},
                    {"text": "from", "start": 1.1, "end": 1.3, "type": "word"},
                    {"text": "ElevenLabs.", "start": 1.4, "end": 2.0, "type": "word"}
                ]
            }"#;

            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                mock_response.len(),
                mock_response
            );
            stream.write_all(response.as_bytes()).unwrap();
        });

        let provider = ElevenLabsProvider::new("test-key-123")
            .with_base_url(format!("http://127.0.0.1:{port}"));

        let cancel = AtomicBool::new(false);
        let req = ProviderRequest {
            audio_bytes: vec![0u8; 100],
            filename: "audio.wav".to_string(),
            mime_type: "audio/wav".to_string(),
            model_id: "scribe_v2".to_string(),
            language: Some("en".to_string()),
            tag_audio_events: false,
            diarize: false,
            num_speakers: None,
        };

        let result = provider.transcribe(&req, &cancel).expect("transcription succeeds");
        assert_eq!(result.text, "Hello world from ElevenLabs");
        assert_eq!(result.words.len(), 4);
        assert_eq!(result.segments.len(), 1);
        assert_eq!(result.segments[0].text, "Hello world from ElevenLabs.");
        assert_eq!(result.segments[0].words.len(), 4);
    }
}
