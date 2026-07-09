# Exp 12c re-analysis — is 'Seneca strongest modifier' robust?

Read-only, from saved circuits. Max node shift = max |Δ normalized_effect|
over nodes shared with base (same metric as the original report).

## Per-item max node shift (sorted by Seneca), Seneca vs Marcus

| item | stance | Seneca | Marcus | winner | margin | tie(<20%) |
|---|---|---|---|---|---|---|
| emot_03 | active | 2.820 | 0.458 | seneca |   516% |  |
| emot_01 | accepting | 1.151 | 0.325 | seneca |   254% |  |
| ext_04 | accepting | 0.535 | 0.450 | seneca |    19% | TIE |
| ctrl_06 | accepting | 0.415 | 0.218 | seneca |    90% |  |
| ctrl_05 | active | 0.405 | 0.109 | seneca |   273% |  |
| emot_02 | active | 0.366 | 0.326 | seneca |    12% | TIE |
| ctrl_03★ | accepting | 0.174 | 0.130 | seneca |    34% |  |
| duty_01★ | active | 0.141 | 0.099 | seneca |    42% |  |
| duty_02 | active | 0.131 | 0.123 | seneca |     7% | TIE |
| mort_03 | accepting | 0.101 | 0.111 | marcus |    10% | TIE |

★ = anchor item (loaded from its own single-item JSON).

## Medians (outlier-robust) vs means

| stat | Seneca | Marcus | Seneca/Marcus |
|---|---|---|---|
| median | 0.386 | 0.174 | 2.22× |
| mean | 0.624 | 0.235 | 2.66× |
| max | 2.820 | 0.458 | — |

Seneca wins 9/10 items; of those, 3 are effective ties (within 20%).

## Effect on content discrimination (Δ|c|) — item-dependent scatter

Across all 10 items Seneca's effect on the content signal is 5 down / 5 up by
direction, with median Δ|c| = +0.06 — no effect on discrimination for a typical
item. The mean is −0.21, dragged negative by two large flattening events
(emot_03 −1.54, ctrl_03 −1.12); removing those two flips the aggregate to
net-sharpening (+0.60). Flatten-vs-sharpen tracks neither base sign, base
magnitude, nor stance. Seneca's effect on the content circuit is item-dependent
scatter around zero — not a directional push and not a uniform disruption —
consistent with 12c's scattered signed Δc. The apparent net flattening is
carried by two high-magnitude outliers, not a global mechanism.

Per-item Δ|c| (|Seneca c| − |base c|), sorted:

| item | stance | base c | Seneca c | Δ\|c\| |
|---|---|---|---|---|
| emot_03 | active | -1.789 | -0.250 | -1.539 |
| ctrl_03 | accepting | +4.484 | +3.367 | -1.117 |
| ext_04 | accepting | -1.766 | -0.883 | -0.883 |
| emot_01 | accepting | -1.508 | -0.742 | -0.766 |
| ctrl_06 | accepting | +1.992 | +1.812 | -0.180 |
| mort_03 | accepting | +1.797 | +2.102 | +0.305 |
| ctrl_05 | active | +2.211 | +2.586 | +0.375 |
| duty_02 | active | -1.711 | -2.141 | +0.430 |
| emot_02 | active | -1.531 | -1.977 | +0.445 |
| duty_01 | active | +2.289 | +3.164 | +0.875 |

ΣΔ|c| = −2.055 · mean −0.205 · median +0.062 · minus {emot_03, ctrl_03} = +0.602.
Among the 5 negative-base-c items (|base c| 1.51–1.79), direction splits both
ways with no ordering by magnitude (emot_01 flatten, emot_02 sharpen, duty_02
sharpen, ext_04 flatten, emot_03 flatten): corr(|base c|, Δ|c|) = −0.27.
