# Changelog
## 0.1.1

### Features

* analyze a frame without enhancing it, and apply the gains to another signal

### Fixes

* write the normalized spectrum in band_unit_norm_t instead of dividing the output buffer

### Performance

* keep the noise suppression's own code free of allocations
