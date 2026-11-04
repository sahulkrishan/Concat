//! Checks `DfTract::delay` against the delay measured on a speech-like signal, which the model
//! keeps (unlike tones or noise).

use ::deep_filter_rt::tract::DfTract;
use ::ndarray::Array2;
use ::std::f32::consts::{PI, TAU};

const SR: f32 = 48000.;

/// A vowel-like signal: a pulse train at 120 Hz through three formant resonators, in syllables of
/// 250 ms.
fn vowel(len: usize) -> Vec<f32> {
    let formants = [(700., 80.), (1220., 90.), (2600., 120.)];
    let mut states = [[0f32; 2]; 3];
    (0..len)
        .map(|i| {
            let pulse = if i % 400 == 0 { 1. } else { 0. };
            let mut y = 0.;
            for (&(freq, bandwidth), s) in formants.iter().zip(states.iter_mut()) {
                let r = (-PI * bandwidth / SR).exp();
                let out = pulse + 2. * r * (TAU * freq / SR).cos() * s[0] - r * r * s[1];
                s[1] = s[0];
                s[0] = out;
                y += out;
            }
            let envelope = (0.5 - 0.5 * (TAU * 4. * i as f32 / SR).cos()).powi(2);
            0.02 * y * envelope
        })
        .collect()
}

/// The lag of `df`'s output against the input with the highest cross-correlation, after the model
/// settled.
fn measured_delay(df: &mut DfTract) -> usize {
    let hop = df.hop_size;
    let input = vowel(200 * hop);
    let mut output = Vec::<f32>::with_capacity(input.len());
    let mut enh = Array2::zeros((1, hop));
    for frame in input.chunks_exact(hop) {
        let noisy = Array2::from_shape_vec((1, hop), frame.to_vec()).unwrap();
        df.process(noisy.view(), enh.view_mut()).unwrap();
        output.extend(enh.iter());
    }

    let max_lag = 2 * df.delay();
    let range = 50 * hop..input.len() - max_lag;
    let corr: Vec<f32> = (0..max_lag)
        .map(|lag| range.clone().map(|i| input[i] * output[i + lag]).sum())
        .collect();
    (0..max_lag)
        .max_by(|&a, &b| corr[a].total_cmp(&corr[b]))
        .unwrap()
}

#[test]
fn delay_matches_the_measured_one() {
    let mut df = DfTract::default();
    assert_eq!(df.delay(), 1440);
    assert_eq!(measured_delay(&mut df), df.delay());
}

#[test]
fn keeps_the_delay_without_attenuation() {
    // an attenuation limit of 0 dB skips the model and passes the input through
    let mut df = DfTract::default();
    df.set_atten_lim(0.);
    assert_eq!(measured_delay(&mut df), df.delay());
}
