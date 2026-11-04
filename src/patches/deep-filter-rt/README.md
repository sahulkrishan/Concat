# deep-filter-rt

**Real-time noise suppression with [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet),
in pure Rust on the CPU.**

It's the inference part of DeepFilterNet's `deep_filter` crate, which is only usable from git (the
last release on crates.io is from 2022). Here it's reduced to the real-time inference, updated to
current dependencies (tract 0.23, ndarray 0.17) and has the DeepFilterNet3 model embedded, so
there's nothing to download. See [NOTICE.md](NOTICE.md) for its origin and the changes.

- full band: 48 kHz audio in frames of 10 ms (480 samples per channel)
- about 2.5 % of a CPU core in release builds
- a delay of 30 ms (`DfTract::delay`: 10 ms STFT overlap plus 2 lookahead frames)
- runs with [tract](https://github.com/sonos/tract), no ONNX Runtime or other native libraries

## Usage

```rust
use deep_filter_rt::tract::DfTract;
use ndarray::Array2;

let mut df = DfTract::default(); // the embedded model with the default parameters
let noisy = Array2::<f32>::zeros((df.ch, df.hop_size));
let mut enhanced = Array2::<f32>::zeros((df.ch, df.hop_size));
let lsnr = df.process(noisy.view(), enhanced.view_mut())?; // local SNR estimate in dB
```

`DfTract::set_atten_lim` limits how much it attenuates, in dB (100, the default, doesn't limit
it); lower values keep more of the original voice. `RuntimeParams` sets it and the other
parameters up front, for `DfTract::new`.

Two variants for more than plain noise suppression:

- `DfTract::analyze` returns only the local SNR estimate of a frame, running the encoder alone
  (about half the time of `process`), e.g. to tell whether a signal has speech.
- `DfTract::process_with` analyzes one signal and applies the gains to another of the same shape
  that contains the same speech, e.g. the output of an echo canceller and the microphone before
  it; the deep filter can be left out, its coefficients are a filter for the analyzed spectrum.

An instance does either `analyze` or `process`/`process_with`, they share its state.

Debug builds run the model about 10 times slower than release builds. The feature `timings` logs
the processing time per frame.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.
