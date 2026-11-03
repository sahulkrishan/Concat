// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! Audio preprocessing, compression/transcoding, and size-limit chunking.
//!
//! Provides utilities to extract audio from media clips into compact mono WAV format,
//! partition oversized audio into overlapping chunks when provider payload limits are
//! exceeded, and stitch chunked transcripts with offset adjustments and boundary deduplication.

use std::sync::atomic::{AtomicBool, Ordering};

use concat_media::{AudioDecoder, AudioOptions, SampleFormat};

use super::{ProviderRequest, SpeechToTextProvider, TranscriptionResult};

/// Target sampling rate for speech recognition (16 kHz mono).
pub const SPEECH_SAMPLE_RATE: u32 = 16_000;

/// Default chunk duration in seconds when splitting large files (10 minutes).
pub const DEFAULT_CHUNK_DURATION_SECS: f64 = 600.0;

/// Overlap duration in seconds between consecutive chunks to avoid cutting words mid-syllable.
pub const CHUNK_OVERLAP_SECS: f64 = 2.0;

/// Audio preprocessor configuration.
#[derive(Clone, Debug)]
pub struct AudioPreprocessor {
    /// Maximum payload size in bytes accepted by the target provider.
    pub max_payload_bytes: usize,
    /// Chunk duration in seconds when splitting is necessary.
    pub chunk_duration_secs: f64,
    /// Overlap in seconds between consecutive chunks.
    pub overlap_secs: f64,
}

impl AudioPreprocessor {
    /// Create a preprocessor configured for a specific provider.
    pub fn for_provider<P: SpeechToTextProvider + ?Sized>(provider: &P) -> Self {
        Self {
            max_payload_bytes: provider.max_payload_bytes(),
            chunk_duration_secs: DEFAULT_CHUNK_DURATION_SECS,
            overlap_secs: CHUNK_OVERLAP_SECS,
        }
    }

    /// Extract 16 kHz mono PCM float samples from a media clip window.
    pub fn extract_mono_samples(
        media_path: &str,
        source_start: f64,
        window: f64,
        audio_stream: Option<u32>,
    ) -> Result<Vec<f32>, String> {
        if window.is_nan() || window <= 0.0 {
            return Err("window duration must be greater than zero".to_owned());
        }

        let mut decoder = AudioDecoder::open(
            media_path,
            &AudioOptions {
                start: Some(source_start),
                duration: Some(window),
                filters: Vec::new(),
                rate: SPEECH_SAMPLE_RATE,
                channels: 1,
                format: SampleFormat::F32,
                stream: audio_stream.map(|s| s as usize),
            },
        )
        .map_err(|e| format!("failed to open audio decoder: {e}"))?;

        let samples = decoder
            .collect_f32()
            .map_err(|e| format!("failed to decode audio frames: {e}"))?;

        if samples.is_empty() {
            return Err("media contains no audio samples in requested window".to_owned());
        }

        Ok(samples)
    }

    /// Convert mono float samples (-1.0..=1.0) to standard 16-bit PCM RIFF WAV bytes.
    pub fn float_samples_to_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        let pcm_16: Vec<i16> = samples
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                (clamped * 32767.0).round() as i16
            })
            .collect();

        pcm16_to_wav(&pcm_16, sample_rate, 1)
    }

    /// Prepare audio chunks for transcription: splits if exceeding size or chunk duration limits.
    pub fn prepare_chunks(&self, samples: &[f32]) -> Vec<AudioChunk> {
        let total_samples = samples.len();
        let total_secs = total_samples as f64 / SPEECH_SAMPLE_RATE as f64;

        let max_samples_by_bytes = self.max_payload_bytes.saturating_sub(44) / 2; // 16-bit mono = 2 bytes/sample
        let max_samples_by_time = (self.chunk_duration_secs * SPEECH_SAMPLE_RATE as f64) as usize;
        let chunk_limit = max_samples_by_bytes.min(max_samples_by_time).max(SPEECH_SAMPLE_RATE as usize);

        // If entire file fits comfortably within limits, return a single chunk
        if total_samples <= chunk_limit {
            let wav_bytes = Self::float_samples_to_wav(samples, SPEECH_SAMPLE_RATE);
            return vec![AudioChunk {
                time_offset: 0.0,
                duration: total_secs,
                wav_bytes,
            }];
        }

        let overlap_samples = (self.overlap_secs * SPEECH_SAMPLE_RATE as f64) as usize;
        let step = chunk_limit.saturating_sub(overlap_samples).max(1);

        let mut chunks = Vec::new();
        let mut start_idx = 0;

        while start_idx < total_samples {
            let end_idx = (start_idx + chunk_limit).min(total_samples);
            let chunk_slice = &samples[start_idx..end_idx];
            let time_offset = start_idx as f64 / SPEECH_SAMPLE_RATE as f64;
            let duration = chunk_slice.len() as f64 / SPEECH_SAMPLE_RATE as f64;
            let wav_bytes = Self::float_samples_to_wav(chunk_slice, SPEECH_SAMPLE_RATE);

            chunks.push(AudioChunk {
                time_offset,
                duration,
                wav_bytes,
            });

            if end_idx >= total_samples {
                break;
            }
            start_idx += step;
        }

        chunks
    }

    /// Transcribe audio chunks with the given provider, adjusting offsets and stitching boundaries.
    pub fn transcribe_and_stitch<P: SpeechToTextProvider + ?Sized>(
        &self,
        provider: &P,
        chunks: &[AudioChunk],
        base_request: &ProviderRequest,
        cancel: &AtomicBool,
    ) -> Result<TranscriptionResult, String> {
        if chunks.is_empty() {
            return Ok(TranscriptionResult::default());
        }

        let mut results = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            if cancel.load(Ordering::Relaxed) {
                return Err("transcription cancelled".to_owned());
            }

            let mut req = base_request.clone();
            req.audio_bytes = chunk.wav_bytes.clone();
            req.filename = "audio.wav".to_owned();
            req.mime_type = "audio/wav".to_owned();

            let mut chunk_res = provider.transcribe(&req, cancel)?;
            // Shift timestamps by chunk offset
            chunk_res.shift_timestamps(chunk.time_offset);
            results.push(chunk_res);
        }

        Ok(stitch_results(results))
    }
}

/// One chunk of audio with its timeline time offset.
#[derive(Clone, Debug)]
pub struct AudioChunk {
    /// Offset in seconds relative to the clip window's start.
    pub time_offset: f64,
    /// Duration of this chunk in seconds.
    pub duration: f64,
    /// RIFF WAV encoded audio payload.
    pub wav_bytes: Vec<u8>,
}

impl TranscriptionResult {
    /// Shift all segment and word timestamps by `offset` seconds.
    pub fn shift_timestamps(&mut self, offset: f64) {
        if offset == 0.0 {
            return;
        }
        for segment in &mut self.segments {
            segment.start += offset;
            segment.end += offset;
            for word in &mut segment.words {
                word.start += offset;
                word.end += offset;
            }
        }
        for word in &mut self.words {
            word.start += offset;
            word.end += offset;
        }
    }
}

/// Encode 16-bit mono or stereo PCM into a standard RIFF/WAVE header and byte buffer.
pub fn pcm16_to_wav(samples: &[i16], sample_rate: u32, num_channels: u16) -> Vec<u8> {
    let subchunk2_size = (samples.len() * 2) as u32;
    let chunk_size = 36 + subchunk2_size;
    let byte_rate = sample_rate * u32::from(num_channels) * 2;
    let block_align = num_channels * 2;

    let mut wav = Vec::with_capacity(44 + subchunk2_size as usize);

    // RIFF chunk descriptor
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&chunk_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // "fmt " sub-chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // Subchunk1Size for PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // AudioFormat (1 = PCM)
    wav.extend_from_slice(&num_channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes()); // BitsPerSample = 16

    // "data" sub-chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&subchunk2_size.to_le_bytes());

    for &sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }

    wav
}

/// Stitch multiple chunked `TranscriptionResult`s into one seamless transcript,
/// removing duplicate words/segments at chunk overlap seams.
pub fn stitch_results(mut chunk_results: Vec<TranscriptionResult>) -> TranscriptionResult {
    if chunk_results.is_empty() {
        return TranscriptionResult::default();
    }
    if chunk_results.len() == 1 {
        return chunk_results.remove(0);
    }

    let mut combined = TranscriptionResult {
        detected_language: chunk_results[0].detected_language.clone(),
        language_probability: chunk_results[0].language_probability,
        ..Default::default()
    };

    let mut full_text_parts = Vec::new();
    let mut last_word_end = -1.0;

    for chunk in chunk_results {
        if !chunk.text.trim().is_empty() {
            full_text_parts.push(chunk.text.trim().to_owned());
        }

        // Deduplicate words that fall into the overlap region of previous chunks
        for word in chunk.words {
            if word.start >= last_word_end - 0.05 {
                last_word_end = word.end;
                combined.words.push(word);
            }
        }

        // Filter and merge subtitle segments
        for seg in chunk.segments {
            if seg.start >= last_word_end - 0.1 || combined.segments.is_empty() {
                combined.segments.push(seg);
            }
        }
    }

    combined.text = full_text_parts.join(" ");
    combined
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm16_to_wav_header() {
        let samples = vec![0i16, 100, -100, 200];
        let wav = pcm16_to_wav(&samples, 16000, 1);

        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
        // 44 bytes header + 4 * 2 = 52 bytes
        assert_eq!(wav.len(), 44 + 8);
    }

    #[test]
    fn test_chunking_under_limit() {
        let preprocessor = AudioPreprocessor {
            max_payload_bytes: 1_000_000,
            chunk_duration_secs: 60.0,
            overlap_secs: 2.0,
        };
        let samples = vec![0.0f32; 16000 * 10]; // 10 seconds
        let chunks = preprocessor.prepare_chunks(&samples);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].time_offset, 0.0);
    }

    #[test]
    fn test_chunking_over_limit() {
        let preprocessor = AudioPreprocessor {
            max_payload_bytes: 100_000, // Small limit forcing chunking
            chunk_duration_secs: 2.0,
            overlap_secs: 0.5,
        };
        let samples = vec![0.0f32; 16000 * 5]; // 5 seconds
        let chunks = preprocessor.prepare_chunks(&samples);
        assert!(chunks.len() >= 3);
        assert_eq!(chunks[0].time_offset, 0.0);
        assert!(chunks[1].time_offset > 0.0);
    }
}
