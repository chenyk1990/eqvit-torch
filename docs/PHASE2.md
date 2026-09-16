# Phase 2: fidelity, Texas training, and latency-aware EEW

## Architecture audit
The released `mseed_predictor.py` was treated as executable specification for the picker and `VIT_Magnitude_Estimation.ipynb` for magnitude. Phase 2 corrects two important details from v0.1: (1) every picker transformer block contains a `convF1` before LayerNorm/MHA and another `convF1` after attention; (2) Keras `MultiHeadAttention(num_heads=4,key_dim=40)` has 160-dimensional internal Q/K/V, unlike `torch.nn.MultiheadAttention(embed_dim=40,num_heads=4)`. `KerasMHA` now reproduces the Keras dimensional convention.

The magnitude network keeps four linear Conv1D blocks (64,32,32,32), pooling (2,2,2,5), 75 samples before ViT, patch=5, projection=100, four heads with key_dim=100, four encoders, and 1000/500 regression MLP. Dropout remains active for Monte-Carlo uncertainty when requested.

## Validation gates
A checkpoint is not called "reproduced" until: architecture shapes match; preprocessing is identical; a fixed waveform produces numerically close TensorFlow and Torch outputs; pick argmax agrees; and batch statistics agree. `scripts/inspect_keras_h5.py` exposes the original HDF5 tensor shapes. The uploaded original picker weights reveal Keras Q/K/V kernels shaped `(40,4,40)`, confirming the 160-wide internal attention representation.

## Texas protocol
Split TXED by **event ID**, never by station waveform. Train magnitude on windows centered on EQCCT-torch predicted P picks (or controlled jitter around manual P when cached predictions are unavailable). Report MAE, RMSE, bias, sigma, residual-vs-M, residual-vs-SNR, station-stratified errors, and event-level bootstrap confidence intervals.

## EEW protocol
Benchmark magnitude at P+2/4/5/10/20/29 s and P-picking at 6000/3000/2000/1000/500 samples. Report accuracy jointly with algorithmic latency and continuous-data false alarms/hour. Short-window picker tests must randomize P position; centering P in every test window leaks the answer.
