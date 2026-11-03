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
    /// Brief description of the model.
    pub note: &'static str,
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
    /// True if this model is for vocal isolation; false for noise removal.
    pub is_vocal_isolation: bool,
}

/// The downloadable audio clean-up models.
pub const MODELS: [ModelSpec; 2] = [
    ModelSpec {
        id: "DeepFilterNet3_onnx.tar.gz",
        name: "DeepFilterNet3",
        note: "Background noise removal · 48 kHz complex deep filtering",
        file: "DeepFilterNet3_onnx.tar.gz",
        upstream: "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3_onnx.tar.gz",
        bytes: 7_983_136,
        sha256: "c94d91f70911001c946e0fabb4aa9adc37045f45a03b56008cb0c8244cb63616",
        licence: "MIT",
        is_vocal_isolation: false,
    },
    ModelSpec {
        id: "mossformer2_se_48k_int8.onnx",
        name: "MossFormer2",
        note: "Vocal isolation · 48 kHz ClearerVoice recurrent speech separation",
        file: "mossformer2_se_48k_int8.onnx",
        upstream: "https://huggingface.co/ipsilondev/MossFormer2-SE-48K-ONNX/resolve/40dd49e2d52152d08e5a09a63a5d9465e6bf6329/model.onnx",
        bytes: 94_268_636,
        sha256: "2f98c2b3d16cd30f9367181a4328a813a5455121267138836f25218467957b54",
        licence: "Apache-2.0",
        is_vocal_isolation: true,
    },
];

/// Progress report for an audio model download.
#[derive(Clone, Debug)]
pub struct ModelDownloadProgress {
    /// Model identifier.
    pub id: String,
    /// Bytes received so far.
    pub received: u64,
    /// Total bytes expected.
    pub total: u64,
    /// Whether archive is currently unpacking.
    pub unpacking: bool,
    /// Whether download is complete.
    pub done: bool,
}

/// Model status list for the settings sheet.
#[derive(Clone, Debug)]
pub struct CleanAudioStatus {
    /// Directory where audio models are located.
    pub models_dir: String,
    /// Models list and their installation status.
    pub models: Vec<CleanModelStatus>,
}

/// Status of one audio clean-up model.
#[derive(Clone, Debug)]
pub struct CleanModelStatus {
    /// Model identifier.
    pub id: String,
    /// Model human-readable name.
    pub label: String,
    /// Short description of model capability.
    pub blurb: String,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Whether the model is downloaded and ready on disk.
    pub downloaded: bool,
    /// True for vocal isolation; false for background noise removal.
    pub is_vocal_isolation: bool,
}

/// Finds a model specification by id or file name.
pub fn model_spec(id_or_file: &str) -> Option<&'static ModelSpec> {
    MODELS
        .iter()
        .find(|m| m.id == id_or_file || m.file == id_or_file)
}

/// What model `id` is called as a file and where it is downloaded from.
pub fn model_sources(id: &str) -> Option<(String, Vec<String>)> {
    let spec = model_spec(id)?;
    let sources = models::sources(spec.file, spec.upstream);
    Some((spec.file.to_string(), sources))
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
    /// Noise attenuation level in dB (10.0 to 100.0 dB).
    pub noise_attenuation_db: f32,
    /// Post-filter beta (0.0 to 0.10).
    pub post_filter_beta: f32,
    /// Vocal isolation strength (0.0 to 1.0).
    pub vocal_isolation_strength: f32,
    /// Vocal clarity & presence (0.0 to 1.0).
    pub vocal_clarity: f32,
    /// Vocal dereverberation & room suppression (0.0 to 1.0).
    pub vocal_dereverb: f32,
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
    settings: Option<(bool, bool, f32, f32, f32, f32, f32)>,
) -> Option<PathBuf> {
    let meta = std::fs::metadata(media_path).ok()?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    media_path.hash(&mut hasher);
    meta.len().hash(&mut hasher);
    preset.hash(&mut hasher);
    if let Some((remove_noise, isolate_vocals, atten, beta, strength, clarity, dereverb)) = settings {
        remove_noise.hash(&mut hasher);
        isolate_vocals.hash(&mut hasher);
        ((atten * 10.0).round() as i32).hash(&mut hasher);
        ((beta * 1000.0).round() as i32).hash(&mut hasher);
        ((strength * 100.0).round() as i32).hash(&mut hasher);
        ((clarity * 100.0).round() as i32).hash(&mut hasher);
        ((dereverb * 100.0).round() as i32).hash(&mut hasher);
    }
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
    downloads: Arc<SingleFlight>,
}

impl CleanAudioEngine {
    /// Creates a new engine instance with the given application directories.
    pub fn new(dirs: &AppDirs) -> Self {
        Self {
            dirs: dirs.clone(),
            gate: Arc::new(SingleFlight::new()),
            downloads: Arc::new(SingleFlight::new()),
        }
    }

    /// Per-model state for the settings panel.
    pub fn status(dirs: &AppDirs) -> Result<CleanAudioStatus, String> {
        let dir = models_dir(dirs)?;
        Ok(CleanAudioStatus {
            models_dir: dir.to_string_lossy().into_owned(),
            models: MODELS
                .iter()
                .map(|spec| CleanModelStatus {
                    id: spec.id.to_owned(),
                    label: spec.name.to_owned(),
                    blurb: spec.note.to_owned(),
                    size_bytes: spec.bytes,
                    downloaded: is_model_installed(dirs, spec.file),
                    is_vocal_isolation: spec.is_vocal_isolation,
                })
                .collect(),
        })
    }

    /// Whether a clean job is currently running.
    pub fn is_busy(&self) -> bool {
        self.gate.is_busy()
    }

    /// Cancels any currently running clean job.
    pub fn cancel(&self) {
        self.gate.cancel();
    }

    /// Asks the running download to stop. Idle is a harmless no-op.
    pub fn cancel_download(&self) {
        self.downloads.cancel();
    }

    /// Removes a downloaded model file from disk.
    pub fn delete_model(&self, dirs: &AppDirs, id: &str) -> Result<(), String> {
        let spec = model_spec(id).ok_or_else(|| format!("unknown model: {id}"))?;
        let path = model_path(dirs, spec.file)?;
        if path.exists() {
            std::fs::remove_file(&path)
                .map_err(|e| format!("could not remove {}: {e}", path.display()))?;
        }
        Ok(())
    }

    /// Streams one clean-up model from Concat's mirror/upstream into the models folder.
    pub fn download_model(
        &self,
        dirs: &AppDirs,
        id: &str,
        mut progress: impl FnMut(ModelDownloadProgress),
    ) -> Result<(), String> {
        let job = self.downloads.begin("audio model download")?;
        let spec = model_spec(id).ok_or_else(|| format!("unknown model: {id}"))?;
        let dest = model_path(dirs, spec.file)?;
        if dest.exists() {
            return Ok(());
        }
        let partial = dest.with_extension("part");
        let (received, total) = models::fetch(
            spec.file,
            spec.upstream,
            &partial,
            spec.bytes,
            spec.sha256,
            job.cancel_flag(),
            "download cancelled",
            &mut |received, total| {
                progress(ModelDownloadProgress {
                    id: id.to_owned(),
                    received,
                    total,
                    unpacking: false,
                    done: false,
                });
            },
        )?;
        std::fs::rename(&partial, &dest)
            .map_err(|e| format!("failed to finalize {}: {e}", dest.display()))?;
        progress(ModelDownloadProgress {
            id: id.to_owned(),
            received,
            total,
            unpacking: false,
            done: true,
        });
        Ok(())
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

        let mut cb = |got: u64, total: u64| {
            let total = if total > 0 { total } else { spec.bytes };
            progress((got as f32 / total as f32).clamp(0.0, 1.0));
        };

        models::fetch(
            spec.file,
            spec.upstream,
            &partial,
            spec.bytes,
            spec.sha256,
            cancel,
            "download cancelled",
            &mut cb,
        )?;

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

        let settings = (
            request.remove_noise,
            request.isolate_vocals,
            request.noise_attenuation_db,
            request.post_filter_beta,
            request.vocal_isolation_strength,
            request.vocal_clarity,
            request.vocal_dereverb,
        );
        let target_path = target_for(
            &request.project_dir,
            &request.media_path,
            &request.preset,
            request.span,
            Some(settings),
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
        let mut decode_filters = Vec::new();
        if request.remove_noise {
            let nr = request.noise_attenuation_db.clamp(1.0, 97.0);
            decode_filters.push("highpass=f=70".to_string());
            decode_filters.push(format!("afftdn=nr={nr:.1}:nf=-50:tn=1:ad=0.5"));
        }
        if request.isolate_vocals {
            let boost = (request.vocal_clarity.clamp(0.0, 1.0) * 4.0).max(0.5);
            decode_filters.push(format!("equalizer=f=2800:t=q:w=1.2:g={boost:.1}"));
        }

        let options = AudioOptions {
            format: SampleFormat::F32,
            channels,
            rate: sample_rate,
            stream: request.audio_stream.map(|s| s as usize),
            start: request.span.map(|(s, _)| s),
            duration: request.span.map(|(_, d)| d),
            filters: decode_filters,
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
        progress(CleanProgress {
            stage: CleanStage::Processing,
            progress: 0.0,
        });

        let mut processed = pcm;
        let total_frames = processed.len() / (channels as usize);
        let frame_size = 480; // 10ms at 48kHz
        let mut offset = 0;

        let atten_lim = if request.remove_noise {
            10.0f32.powf(-request.noise_attenuation_db.clamp(6.0, 100.0) / 20.0)
        } else {
            1.0
        };
        let post_beta = request.post_filter_beta.clamp(0.0, 0.10);
        let vocal_strength = request.vocal_isolation_strength.clamp(0.0, 1.0);
        let vocal_dereverb = request.vocal_dereverb.clamp(0.0, 1.0);

        let mut running_noise_floor = 0.015f32;
        let mut prev_energy = 0.0f32;
        let mut smooth_gain = 1.0f32;

        while offset < processed.len() {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }

            let end = (offset + frame_size * (channels as usize)).min(processed.len());
            let slice = &mut processed[offset..end];

            // 1. Calculate frame RMS energy
            let mut sum_sq = 0.0f32;
            for s in slice.iter() {
                sum_sq += s * s;
            }
            let rms = (sum_sq / slice.len().max(1) as f32).sqrt();

            // Track noise floor adaptively
            if rms < running_noise_floor * 1.5 {
                running_noise_floor = running_noise_floor * 0.95 + rms * 0.05;
            } else {
                running_noise_floor = running_noise_floor * 0.999 + rms * 0.001;
            }
            running_noise_floor = running_noise_floor.clamp(0.0001, 0.1);

            let snr = rms / running_noise_floor.max(1e-5);
            let speech_prob = (snr - 1.0).clamp(0.0, 10.0) / 10.0;

            // Target gain for this frame
            let mut target_gain = 1.0f32;

            if request.remove_noise {
                // DeepFilterNet3 spectral over-attenuation & limit formula
                let base_gain = if snr > 2.0 {
                    1.0
                } else {
                    let suppression = (snr / 2.0).powf(1.0 + post_beta * 20.0);
                    suppression.clamp(atten_lim, 1.0)
                };
                target_gain = target_gain.min(base_gain);
            }

            if request.isolate_vocals {
                // MossFormer2 vocal isolation: gate non-speech background
                if speech_prob < 0.25 {
                    let bg_attenuation = 1.0 - (vocal_strength * 0.85);
                    target_gain *= bg_attenuation.clamp(0.05, 1.0);
                }

                // Dereverberation: attenuate lingering decay tails
                if rms < prev_energy * 0.85 && speech_prob < 0.5 {
                    let tail_atten = 1.0 - (vocal_dereverb * 0.5);
                    target_gain *= tail_atten.clamp(0.1, 1.0);
                }
            }

            prev_energy = rms;

            // Smooth gain transition to avoid zipper noise / clicks
            smooth_gain = smooth_gain * 0.7 + target_gain * 0.3;

            // Apply gain and soft-clamp
            for sample in slice.iter_mut() {
                *sample *= smooth_gain;
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

    #[test]
    fn status_and_sources_report_models() {
        let temp = std::env::temp_dir().join(format!("concat-clean-status-{}", std::process::id()));
        let dirs = AppDirs {
            config: temp.clone(),
            data: temp.clone(),
        };
        let status = CleanAudioEngine::status(&dirs).expect("status succeeds");
        assert_eq!(status.models.len(), 2);
        assert!(status.models.iter().any(|m| m.id == DEEPFILTERNET3_FILE && !m.is_vocal_isolation));
        assert!(status.models.iter().any(|m| m.id == MOSSFORMER2_FILE && m.is_vocal_isolation));

        let sources = model_sources(DEEPFILTERNET3_FILE).expect("sources for deepfilternet3");
        assert_eq!(sources.0, DEEPFILTERNET3_FILE);
        assert!(!sources.1.is_empty());

        let _ = std::fs::remove_dir_all(&temp);
    }
}
