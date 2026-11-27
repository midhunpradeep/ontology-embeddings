# TOWN-02 shared-training profile

Reproduce with the project environment (run each phase against its corresponding source state):

```bash
PYTHONPATH=src /path/to/python evidence/town02_shared_training_profile.py \
  --phase pre --output evidence/town02_profile_pre.json
PYTHONPATH=src /path/to/python evidence/town02_shared_training_profile.py \
  --phase post --output evidence/town02_profile_post.json
```

Both profiles use Python 3.12.0, PyTorch 2.6.0+cpu, one CPU thread, fixed-seed synthetic inputs, bounded BOX H=1, latent dimension 16, 60 classes, and a batch of 64 pizzas / 256 component occurrences (23 pizza features, 22 component features, 320 points total). This is representative in shape only; no real dataset was read. JSON records per-stage samples/medians and source SHA-256 fingerprints. The pre fingerprint is the clean base `d647fcc50576969f53fd6fe2b1ac1597fadf57d4`; post fingerprints identify the uncommitted working-tree sources.

| Stage (median ms) | Pre | Post |
|---|---:|---:|
| Pack records | 9.72 | 9.11 |
| Encode | 0.238 | 0.232 |
| Decode + score + backward | 10.32 | 10.12 |
| Masked loss + backward | 100.35 | 1.67 |
| Evaluation + diagnostics | 114.30 | 106.31 |
| Optimizer-state checks | 0.00336 | 0.00333 |
| Atomic checkpoint persistence | 1.43 | 1.51 |
| Complete loss path with predictions | 83.73 | 17.13 |

The material measured bottleneck was row-by-row masked reduction (about 98.3% faster at this shape after vectorizing per-entity and per-parent means). The complete loss path improved about 79.5%; `_box_loss` now encodes pizza/component features once each, decodes endpoints once, computes score and (only when requested) exact membership from those same points/endpoints, and skips prediction entirely on update-only passes. Independent Boolean membership remains authoritative; no clearance-sign shortcut was introduced. Unit parity covers exact membership/clearance, loss, gradients, and an Adam update.

Packing, standalone encoding, score/backward, optimizer checks, and checkpoint timings changed little or within run noise; no separate optimizations were made to them. Evaluation/diagnostics still costs about 95 ms and is left unchanged rather than expanding scope. CUDA was unavailable, so peak GPU allocation is `null`. Process high-water RSS was about 303–305 MiB, cumulative and not stage-isolated; it is not evidence of a meaningful memory change.
