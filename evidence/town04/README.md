# TOWN-04 compact evidence

Checksummed publication for TOWN-SPEC #21 and implementation issues #23–#26, using existing producer artifacts only. No raw dataset or private paths are included. The original run source revision is [`3714b8f1e7e1aa94ab252734c6a73b251973077f`](https://github.com/midhunpradeep/ontology-embeddings/commit/3714b8f1e7e1aa94ab252734c6a73b251973077f), clean for both runs. Machine-readable metrics, diagnostics, identities and hashes are in [`summary.json`](summary.json); signature strata are in [`validation-signatures.json`](validation-signatures.json). Selected checkpoint copies are included and retain the source artifact hashes.

## Protocol and identity

Dataset `pizza-6a33710d13eb90fdde3e`; 720 train and 240 validation records, no test records loaded or evaluated. Both runs used seed 23, d=16, H=1/W=1, logistic τ=0.1, 60 epochs, batch size 64, Adam LR 0.02, float32, 720 optimizer updates. Unbounded and bounded runs share paired initialization; independently reviewed cross-domain physical frame/window coordinate discrepancy was at most `2.3842e-7`, below `2e-5`. This establishes physical coordinate matching, not equality of raw parameterizations or Jacobians.

## Selected validation results

| Domain / run | Best epoch | Pizza micro-F1 (train-frequency baseline) | Component micro-F1 (baseline) | Runtime / wall | Peak CUDA allocated |
|---|---:|---:|---:|---:|---:|
| Unbounded `20261127T142303-452045129f33` | 49 | 0.986715 (0.647554) | 1.000000 (0.731985) | 113.10 / 115.51 s | 338,973,184 B |
| Bounded `20261127T142545-2e2b89819b9f` | 45 | 0.987311 (0.647554) | 0.924005 (0.731985) | 114.30 / 116.78 s | 339,833,856 B |

Both tasks exceeded the respective train-only frequency baseline at each selected checkpoint. Best-checkpoint reload predictions were equivalent. The selected checkpoints are `selected-unbounded-best.pt` (`686b8223…`) and `selected-bounded-best.pt` (`13793839…`).

## Learning and geometry diagnostics

All four frozen fixed-coordinate controls passed: unbounded/bounded × logistic/hinge, with target recovery, finite useful gradients and optimizer movement. At both selected epochs, pizza and component adapters and frame/window geometry had nonzero reported gradient norms, parameter movement, and updates.

At the selected checkpoint, unbounded frames had width range 0.8412–4.8904 and observed closed-frame membership fraction 0.1675; bounded frames had width range 0.5899–1.9839 and fraction 0.5797. The one window per class was contained in one class in each domain; independent windows were allowed to protrude. Unbounded geometric host-frame exclusion fractions ranged 0–0.5809 (median 0.2055), and observed frame-point exclusion fractions 0–0.9726 (median 0.0403); observed TOWN membership exclusion was 0.0427. Bounded values were 0.0001–0.6248 (median 0.2734), 0–0.9997 (median 0.7334), and 0.4330 respectively. These diagnostics distinguish geometric exclusion from observed membership contribution; they are not global ontology-fidelity claims.

Signature-stratified validation: pizza had 143 seen and 97 unseen entities; unseen pizza micro-F1 was 0.966868 unbounded and 0.968326 bounded. Components had 4,028 seen and zero unseen entities; the unseen component score is undefined. See the JSON for both seen scores and counts.

## Evidence correction

[`CORRECTION.md`](CORRECTION.md) and [`correction-record.json`](correction-record.json) reconcile the epoch-60 `final.pt` metadata against the exact final records in retained producer history, with original checkpoint and history hashes. The producer fix is source revision [`aff5e888b7206af5be45ac14a26254af5ad26487`](https://github.com/midhunpradeep/ontology-embeddings/commit/aff5e888b7206af5be45ac14a26254af5ad26487). Original artifacts were not rewritten; no derivative checkpoint or retraining was needed. The discrepancy is metadata-only and does not affect selected best checkpoints, measured metrics, geometry, or scientific conclusions.

## Dispositions and limitations

Prior independent scientific dispositions: A PASS, B PASS, C PASS. A fresh targeted independent review returned PASS on the producer correction, regression, historical values, original-artifact provenance, and the evidence-integrity finding. The evidence is a preliminary single-seed, validation-only demonstration, not a sealed-test result, external-generalization claim, or definitive comparative conclusion.

All files in this directory (except `SHA256SUMS` itself) are listed with SHA-256 in [`SHA256SUMS`](SHA256SUMS).
