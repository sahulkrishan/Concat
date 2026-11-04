# Origin

This crate is a vendored copy of `deep_filter` (folder `libDF`) from
[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) by Hendrik Schröter, taken without its
history at commit
[`d375b2d`](https://github.com/Rikorose/DeepFilterNet/tree/d375b2d8309e0935d165700c91da9de862a99c31)
(2024-10-17). It's licensed under MIT or Apache-2.0, like the original, see `LICENSE`,
`LICENSE-MIT` and `LICENSE-APACHE`.

The DeepFilterNet3 model in `models/DeepFilterNet3/` is the content of upstream's
`models/DeepFilterNet3_onnx.tar.gz` at the same commit, unpacked and otherwise unchanged.

## Changes

`src/lib.rs` and `src/tract.rs` are modified, all other source files of upstream are removed:

- renamed from `deep_filter` (library `df`) to `deep-filter-rt`, the version starts again at 0.1.0
- reduced to the real-time inference: the training, dataset and augmentation code, the transforms,
  the C API, the WASM bindings, the logging bridge and the binaries are removed, and with them all
  features but `timings`
- the model is embedded from its unpacked files: `DfParams::new` takes a model directory and
  `DfParams::from_bytes` the config and the three ONNX files instead of a `tar.gz` archive
- updated to tract 0.23 (from 0.21), ndarray 0.17 (from 0.15) and itertools 0.15, edition 2024
- unused functions removed, clippy and rustfmt fixes, typos in comments fixed
- the inline unit tests are replaced by `tests/model.rs`

The changes after this are in the history of the repository.
