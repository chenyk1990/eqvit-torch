# EQViT-torch
Real-Time Earthquake Detection and Magnitude Estimation using Vision Transformer

A PyTorch reimplementation and Texas-oriented extension of **EQViT**, integrating **P-wave arrival picking** with **rapid single-station magnitude estimation**. The package keeps the original 60-s/6000-sample picker and 30-s/3000-sample magnitude concepts, while adding TXED-native training, EQCCT-torch interoperability hooks, Monte-Carlo-dropout uncertainty, streaming utilities, and a research path toward 30/10/<10-s P-only pickers.


## Reference

Saad, Omar M., Yunfeng Chen, Alexandros Savvaidis, Sergey Fomel, and Yangkang Chen. "Real‐Time Earthquake Detection and Magnitude Estimation Using Vision Transformer." Journal of Geophysical Research: Solid Earth 127, no. 5 (2022): e2021JB023657.

BibTeX:

      @article{eqvit,
        title={Real-Time Earthquake Detection and Magnitude Estimation Using Vision Transformer},
        author={Saad, Omar M and Chen, Yunfeng and Savvaidis, Alexandros and Fomel, Sergey and Chen, Yangkang},
        journal={Journal of Geophysical Research: Solid Earth},
        volume={127},
        number={5},
        pages={e2021JB023657},
        year={2022},
        publisher={Wiley Online Library}
      }
-----------

## Install
```bash
pip install -e .
# notebooks
pip install -e '.[notebook]'
```

## Core workflows
- `01_original_eqvit_torch_inference.ipynb`: architecture + Torch checkpoint loading + picking/magnitude inference.
- `02_txed_magnitude_training.ipynb`: event-disjoint TXED training using `magnitude` and `p_arrival_sample`.
- `03_eqcct_txed_picks_to_magnitude.ipynb`: production training concept: obtain imperfect P picks from EQCCT-torch, then train magnitude on windows centered on those picks.
- `04_texnet_streaming_eew.ipynb`: continuous/streaming waveform -> P pick -> magnitude -> latency-oriented output.
- `05_short_window_p_picker_research.ipynb`: 6000 -> 3000 -> 1000 -> 500-sample P-only experiments.

## Important amplitude rule
Picker input is standardized; magnitude input is **not amplitude-normalized**. This mirrors the scientific requirement that absolute amplitude carries magnitude information. De-meaning is allowed; arbitrary per-window standardization is not.

## Existing Torch model
Download the user-specified Figshare Torch artifact (`file=31189627`) and place it under `weights/`. `eqvit_torch.weights.load_legacy_torch_model` handles a serialized `nn.Module`; `load_checkpoint` handles common state-dict checkpoints. Because external checkpoint serialization/key conventions can differ, the first notebook prints a compatibility report rather than silently accepting mismatched keys.

## TXED
Set `TXED_H5` and `TXED_IDS` to your local `TXED_20231111.h5` and `ID_20231111.npy`. Splits are by **event**, not trace, to avoid leakage between stations recording the same earthquake.

## EEW design note
The 30-s magnitude model is a reproducible baseline, not the final latency target. For EEW, train/evaluate magnitude models at progressively shorter post-P durations (e.g. 20, 10, 5, 4, 2 s), reporting MAE versus latency and uncertainty. The original study found useful results with short windows but degradation when windows become too short.

## Phase 2 (v0.2)

Phase 2 audits the released Keras source rather than relying only on the paper diagram. The PyTorch picker now reproduces the released transformer's `convF1 → LayerNorm → MHA → convF1 → residual → LayerNorm → MLP → residual` structure and Keras `key_dim` semantics. The magnitude model follows the released 3000×3 notebook architecture. See `docs/PHASE2.md`.

### Recommended Texas workflow

```bash
pip install -e .
python scripts/inspect_keras_h5.py /path/to/test_trainer_011.h5
jupyter lab notebooks/02_txed_magnitude_training.ipynb
```

Use TXED's `TXED_20231111.h5` + `ID_20231111.npy`. The official TXED examples store each waveform under its waveform ID, with `data` and attributes such as `p_arrival_sample`, `magnitude`, and `snr_db`. For operationally realistic magnitude training, generate an EQCCT-torch P-pick cache and continue with notebook 03.

### Phase-2 notebooks

- `02_txed_magnitude_training.ipynb`: event-disjoint magnitude baseline and diagnostics.
- `03_eqcct_txed_picks_to_magnitude.ipynb`: magnitude training around EQCCT-torch predicted P arrivals.
- `04_texnet_streaming_eew.ipynb`: streaming contract, MC-dropout uncertainty, and latency-aware updates.
- `05_short_window_p_picker_research.ipynb`: 60/30/20/10/5-s P-picker benchmark without P-position leakage.

### Important validation rule

A Keras checkpoint is not declared numerically ported merely because PyTorch tensor shapes load. Exact equivalence requires the same preprocessing plus waveform-level TensorFlow-vs-PyTorch output comparisons. The package keeps this distinction explicit.


## Phase 2.1: main TXED magnitude-training notebook

Start here for magnitude training:

`notebooks/02_TXED_train_robust_EQViT_magnitude.ipynb`

This is the complete TXED workflow: local HDF5 audit, metadata QC, event-disjoint splitting,
amplitude-preserving P-centered windows, optional magnitude-balanced sampling, robust training,
checkpointing/early stopping, locked test evaluation, magnitude/SNR/station diagnostics,
P-pick-error stress testing, worst-case waveform inspection, and event-level multi-station fusion.

Set `TXED_DIR` in the configuration cell, run first with `QUICK_RUN=True`, then use
`QUICK_RUN=False` for the full experiment.
