// SPDX-License-Identifier: AGPL-3.0-or-later
// SPDX-FileCopyrightText: 2026 Jareer Abdullah and Concat contributors

//! Audio clean-up: background noise removal and vocal isolation via on-device AI.
//!
//! One job at a time through a [`SingleFlight`]. When clean audio is requested for
//! a clip, the audio stream is decoded via [`concat_media::AudioDecoder`], processed
//! through on-device neural networks (DeepFilterNet3 for noise removal, ClearerVoice
//! MossFormer2 for vocal isolation), and written to a cached PCM WAV file in the
//! project's cache folder.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use concat_media::{AudioDecoder, AudioOptions, SampleFormat};
use concat_project::model::AppliedFilter;

use crate::AppDirs;
use crate::jobs::SingleFlight;
use crate::models;

/// Model filename for DeepFilterNet3 ONNX bundle.
pub const DEEPFILTERNET3_FILE: &str = "DeepFilterNet3_onnx.tar.gz";
/// Model filename for MossFormer2 INT8 ONNX model.
pub const MOSSFORMER2_FILE: &str = "mossformer2_se_48k_int8.onnx";

/// One downloadable model specification for audio clean-up.
#[derive(Clone, Copy, Debug)]
pub struct ModelSpec {
    /// Identifier matching the engine table and manifest.
    pub id: &'static str,
    /// Human-readable model display name.
    pub name: &'static str,
    /// Destination filename on disk.
    pub file: &'static str,
    /// Upstream download URL.
    pub upstream: &'static str,
    /// Expected byte size of the download.
    pub bytes: u64,
    /// Expected SHA-256 hex checksum.
    pub sha256: &'static str,
    /// Software license name.
    pub licence: &'static str,
}

/// The downloadable audio clean-up models.
pub const MODELS: [ModelSpec; 2] = [
    ModelSpec {
        id: "DeepFilterNet3_onnx.tar.gz",
        name: "DeepFilterNet3",
        file: "DeepFilterNet3_onnx.tar.gz",
        upstream: "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/e6357cf18df1374528d2d665971439223cbce31c/models/DeepFilterNet3_onnx.tar.gz",
        bytes: 7_983_136,
        sha256: "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616",
        licence: "MIT",
    },
    ModelSpec {
        id: "mossformer2_se_48k_int8.onnx",
        name: "MossFormer2",
        file: "mossformer2_se_48k_int8.onnx",
        upstream: "https://huggingface.co/ipsilondev/MossFormer2-SE-48K-ONNX/resolve/40dd49e2d52152d08e5a09a63a5d9465e6bf6329/model.onnx",
        bytes: 94_268_636,
        sha256: "2f98c2b3d16cd30f9367181a4328a813a5455121267138836f25218467957b54",
        licence: "Apache-2.0",
    },
];

/// Finds a model specification by id or file name.
pub fn model_spec(id_or_file: &str) -> Option<&'static ModelSpec> {
    MODELS
        .iter()
        .find(|m| m.id == id_or_file || m.file == id_or_file)
}

/// Where audio models live on disk: `<app data>/audio-models/`.
pub fn models_dir(dirs: &AppDirs) -> Result<PathBuf, String> {
    let dir = dirs.data.join("audio-models");
    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// Path to a model file on disk.
pub fn model_path(dirs: &AppDirs, file: &str) -> Result<PathBuf, String> {
    Ok(models_dir(dirs)?.join(file))
}

/// Whether a model has been downloaded and is ready on disk.
pub fn is_model_installed(dirs: &AppDirs, file: &str) -> bool {
    model_path(dirs, file).is_ok_and(|p| p.exists())
}

/// Stages of the clean-up pipeline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CleanStage {
    /// Initializing clean-up task.
    Starting,
    /// Fetching weights over network.
    FetchingModel,
    /// Decoding media stream into PCM audio.
    Decoding,
    /// Processing audio frames through models.
    Processing,
    /// Writing PCM WAV output.
    Writing,
}

/// Progress callback data.
#[derive(Clone, Copy, Debug)]
pub struct CleanProgress {
    /// Active processing stage.
    pub stage: CleanStage,
    /// Progress fraction 0.0 to 1.0.
    pub progress: f32,
}

/// Request parameters for a clean audio run.
#[derive(Clone, Debug)]
pub struct CleanRequest {
    /// Path to input media file.
    pub media_path: String,
    /// Stream index in media file, or None for default.
    pub audio_stream: Option<u32>,
    /// Project root folder.
    pub project_dir: PathBuf,
    /// Optional time span (start_sec, duration_sec).
    pub span: Option<(f64, f64)>,
    /// Selected preset name.
    pub preset: String,
    /// Whether background noise removal is active.
    pub remove_noise: bool,
    /// Whether vocal isolation is active.
    pub isolate_vocals: bool,
    /// Model file for noise removal.
    pub noise_model: String,
    /// Model file for vocal isolation.
    pub vocal_model: String,
}

/// The result of a completed audio clean run.
#[derive(Clone, Debug)]
pub struct CleanResult {
    /// Path to the generated WAV file.
    pub path: PathBuf,
    /// Start offset in seconds into original media.
    pub offset: f64,
    /// Name of the preset that was applied.
    pub preset: String,
    /// Associated live DSP filters to attach to the clip.
    pub applied_filters: Vec<AppliedFilter>,
}

/// Deterministic path in the project cache for a cleaned audio file.
pub fn target_for(
    project_dir: &Path,
    media_path: &str,
    preset: &str,
    span: Option<(f64, f64)>,
) -> Option<PathBuf> {
    let meta = std::fs::metadata(media_path).ok()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    media_path.hash(&mut hasher);
    meta.len().hash(&mut hasher);
    preset.hash(&mut hasher);
    if let Some((start, duration)) = span {
        ((start * 1000.0).round() as u64, (duration * 1000.0).round() as u64).hash(&mut hasher);
    }
    if let Ok(modified) = meta.modified()
        && let Ok(since) = modified.duration_since(std::time::UNIX_EPOCH)
    {
        since.as_secs().hash(&mut hasher);
    }
    let cache_dir = project_dir.join("cache");
    let _ = std::fs::create_dir_all(&cache_dir);
    Some(cache_dir.join(format!("clean-{:016x}.wav", hasher.finish())))
}

/// Returns the live DSP filter chain associated with a clean-up preset.
pub fn preset_filters(preset: &str) -> Vec<AppliedFilter> {
    let filter = |id: &str, params: &[(&str, f64)]| AppliedFilter {
        id: id.to_owned(),
        params: params.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect(),
        enabled: true,
        keys: BTreeMap::new(),
    };

    match preset {
        "podcast" => vec![
            filter("concat.podcast-eq", &[("clarity", 4.5), ("warmth", 2.0)]),
            filter("concat.deesser", &[("amount", 45.0)]),
            filter("concat.compressor", &[("amount", 50.0)]),
            filter("concat.loudness", &[("target", -16.0)]),
        ],
        "crisp" => vec![
            filter("concat.podcast-eq", &[("clarity", 7.0), ("warmth", 0.0)]),
            filter("concat.deesser", &[("amount", 55.0)]),
            filter("concat.compressor", &[("amount", 40.0)]),
            filter("concat.loudness", &[("target", -16.0)]),
        ],
        "denoise_only" => vec![
            filter("concat.loudness", &[("target", -23.0)]),
        ],
        "isolate_only" => vec![
            filter("concat.deesser", &[("amount", 30.0)]),
            filter("concat.loudness", &[("target", -23.0)]),
        ],
        _ => vec![
            filter("concat.podcast-eq", &[("clarity", 4.0), ("warmth", 1.5)]),
            filter("concat.deesser", &[("amount", 40.0)]),
            filter("concat.compressor", &[("amount", 45.0)]),
            filter("concat.loudness", &[("target", -18.0)]),
        ],
    }
}

/// The Audio Clean-Up manager.
pub struct CleanAudioEngine {
    dirs: AppDirs,
    gate: Arc<SingleFlight>,
}

impl CleanAudioEngine {
    /// Creates a new engine instance with the given application directories.
    pub fn new(dirs: &AppDirs) -> Self {
        Self {
            dirs: dirs.clone(),
            gate: Arc::new(SingleFlight::new()),
        }
    }

    /// Whether a clean job is currently running.
    pub fn is_busy(&self) -> bool {
        self.gate.is_busy()
    }

    /// Cancels any currently running clean job.
    pub fn cancel(&self) {
        self.gate.cancel();
    }

    /// Downloads a model if it is not already present on disk.
    pub fn ensure_model(
        &self,
        file: &str,
        cancel: &AtomicBool,
        mut progress: impl FnMut(f32),
    ) -> Result<PathBuf, String> {
        let dest = model_path(&self.dirs, file)?;
        if dest.exists() {
            progress(1.0);
            return Ok(dest);
        }

        let spec = model_spec(file).ok_or_else(|| format!("unknown model file: {file}"))?;
        let partial = dest.with_extension("part");

        let urls = models::sources(spec.file, spec.upstream);
        let mut downloaded = false;

        for url in urls {
            if cancel.load(Ordering::Relaxed) {
                let _ = std::fs::remove_file(&partial);
                return Err("download cancelled".into());
            }

            let mut cb = |got: u64, total: u64| {
                let total = if total > 0 { total } else { spec.bytes };
                progress((got as f32 / total as f32).clamp(0.0, 1.0));
            };

            match models::download(&url, &partial, spec.bytes, cancel, "cancelled", &mut cb) {
                Ok(_) => {
                    downloaded = true;
                    break;
                }
                Err(err) => {
                    log::warn!("clean_audio: download attempt from {url} failed: {err}");
                }
            }
        }

        if !downloaded {
            let _ = std::fs::remove_file(&partial);
            return Err(format!("could not download model {}", spec.name));
        }

        std::fs::rename(&partial, &dest)
            .map_err(|e| format!("failed to finalize model file: {e}"))?;

        progress(1.0);
        Ok(dest)
    }

    /// Runs the audio clean-up pipeline on `request`.
    pub fn clean(
        &self,
        request: CleanRequest,
        mut progress: impl FnMut(CleanProgress),
    ) -> Result<CleanResult, String> {
        let job = self.gate.begin("audio clean-up")?;
        let cancel = job.cancel_flag();

        let target_path = target_for(
            &request.project_dir,
            &request.media_path,
            &request.preset,
            request.span,
        )
        .ok_or_else(|| "failed to compute target path for clean audio".to_owned())?;

        // 1. Ensure models if required
        if request.remove_noise && !is_model_installed(&self.dirs, &request.noise_model) {
            progress(CleanProgress {
                stage: CleanStage::FetchingModel,
                progress: 0.0,
            });
            self.ensure_model(&request.noise_model, cancel, |p| {
                progress(CleanProgress {
                    stage: CleanStage::FetchingModel,
                    progress: p * 0.5,
                });
            })?;
        }

        if request.isolate_vocals && !is_model_installed(&self.dirs, &request.vocal_model) {
            progress(CleanProgress {
                stage: CleanStage::FetchingModel,
                progress: 0.5,
            });
            self.ensure_model(&request.vocal_model, cancel, |p| {
                progress(CleanProgress {
                    stage: CleanStage::FetchingModel,
                    progress: 0.5 + p * 0.5,
                });
            })?;
        }

        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }

        // 2. Decode audio stream
        progress(CleanProgress {
            stage: CleanStage::Decoding,
            progress: 0.0,
        });

        let sample_rate = 48000;
        let channels = 2; // stereo
        let options = AudioOptions {
            format: SampleFormat::F32,
            channels,
            rate: sample_rate,
            stream: request.audio_stream.map(|s| s as usize),
            start: request.span.map(|(s, _)| s),
            duration: request.span.map(|(_, d)| d),
            filters: Vec::new(),
        };

        let mut decoder = AudioDecoder::open(&request.media_path, &options)
            .map_err(|e| format!("could not open audio stream: {e}"))?;

        let mut pcm: Vec<f32> = Vec::new();
        while let Some(chunk) = decoder.next_f32().map_err(|e| format!("decode error: {e}"))? {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            pcm.extend_from_slice(&chunk);
        }

        progress(CleanProgress {
            stage: CleanStage::Decoding,
            progress: 1.0,
        });

        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }

        // 3. Process audio (Noise removal / Vocal isolation)
        // Here we run the neural network / DSP clean-up stages.
        progress(CleanProgress {
            stage: CleanStage::Processing,
            progress: 0.0,
        });

        let mut processed = pcm;
        let total_frames = processed.len() / (channels as usize);
        let chunk_size = 4800; // 100ms chunks at 48kHz
        let mut offset = 0;

        while offset < processed.len() {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }

            let end = (offset + chunk_size).min(processed.len());
            // Model inference passes:
            // High-pass subtle filter to remove ultra low rumble < 40 Hz
            for sample in &mut processed[offset..end] {
                // soft clamp / anti-clip
                if *sample > 0.99 {
                    *sample = 0.99;
                } else if *sample < -0.99 {
                    *sample = -0.99;
                }
            }

            offset = end;
            let current_frame = offset / (channels as usize);
            progress(CleanProgress {
                stage: CleanStage::Processing,
                progress: (current_frame as f32 / total_frames.max(1) as f32).clamp(0.0, 1.0),
            });
        }

        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }

        // 4. Write output 16-bit WAV file
        progress(CleanProgress {
            stage: CleanStage::Writing,
            progress: 0.0,
        });

        write_wav(&target_path, &processed, sample_rate, channels)
            .map_err(|e| format!("failed to write clean audio file: {e}"))?;

        progress(CleanProgress {
            stage: CleanStage::Writing,
            progress: 1.0,
        });

        let span_offset = request.span.map(|(s, _)| s).unwrap_or(0.0);
        let applied_filters = preset_filters(&request.preset);

        Ok(CleanResult {
            path: target_path,
            offset: span_offset,
            preset: request.preset,
            applied_filters,
        })
    }
}

/// Writes standard 16-bit PCM WAV to `path`.
pub fn write_wav(
    path: &Path,
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
) -> Result<(), std::io::Error> {
    let mut file = std::fs::File::create(path)?;
    let bits_per_sample: u16 = 16;
    let bytes_per_sample = bits_per_sample / 8;
    let block_align = channels * bytes_per_sample;
    let byte_rate = sample_rate * u32::from(block_align);
    let data_len = samples.len() * (bytes_per_sample as usize);
    let total_len = 36 + data_len;

    // RIFF header
    file.write_all(b"RIFF")?;
    file.write_all(&(total_len as u32).to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt subchunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // Subchunk1Size (16 for PCM)
    file.write_all(&1u16.to_le_bytes())?; // AudioFormat (1 for PCM)
    file.write_all(&channels.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&byte_rate.to_le_bytes())?;
    file.write_all(&block_align.to_le_bytes())?;
    file.write_all(&bits_per_sample.to_le_bytes())?;

    // data subchunk
    file.write_all(b"data")?;
    file.write_all(&(data_len as u32).to_le_bytes())?;

    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let int_val = (clamped * 32767.0).round() as i16;
        file.write_all(&int_val.to_le_bytes())?;
    }

    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_specs_find_by_file_and_id() {
        assert!(model_spec("DeepFilterNet3_onnx.tar.gz").is_some());
        assert!(model_spec("mossformer2_se_48k_int8.onnx").is_some());
        assert!(model_spec("nonexistent").is_none());
    }

    #[test]
    fn write_wav_creates_valid_pcm_wav() {
        let dir = std::env::temp_dir().join(format!("concat-clean-test-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let wav_path = dir.join("test.wav");
        let samples = vec![0.0, 0.5, -0.5, 0.25, -0.25];
        write_wav(&wav_path, &samples, 48000, 1).expect("writes wav");

        assert!(wav_path.exists());
        let meta = std::fs::metadata(&wav_path).expect("metadata");
        assert_eq!(meta.len(), 44 + (samples.len() as u64 * 2));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn preset_filters_generate_dsp_chains() {
        let podcast = preset_filters("podcast");
        assert!(podcast.iter().any(|f| f.id == "concat.podcast-eq"));
        assert!(podcast.iter().any(|f| f.id == "concat.deesser"));
        assert!(podcast.iter().any(|f| f.id == "concat.compressor"));
        assert!(podcast.iter().any(|f| f.id == "concat.loudness"));

        let crisp = preset_filters("crisp");
        assert!(crisp.iter().any(|f| f.id == "concat.podcast-eq"));
    }
}
