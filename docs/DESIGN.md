# Design decisions

1. **Two-stage interface remains explicit.** A P picker and a magnitude regressor are independently replaceable.
2. **TXED event-disjoint splits.** Multiple stations for one earthquake must not cross train/test boundaries.
3. **Deployment-error training.** Magnitude training should preferentially center on predicted EQCCT-torch P picks, not only manual picks.
4. **Amplitude preservation.** Never per-window standardize magnitude input unless the removed scale is explicitly reintroduced.
5. **Uncertainty.** MC dropout is available for compatibility/research; calibration should be validated on held-out Texas events.
6. **Streaming.** Trigger de-duplication, station QC, multi-station association and alert policy are service-layer responsibilities.
7. **Short windows.** Treat picker context length and magnitude post-P duration as separate axes in benchmarks.
