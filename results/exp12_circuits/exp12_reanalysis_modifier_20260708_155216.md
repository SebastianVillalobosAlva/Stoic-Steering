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

## Top-3 Seneca effects — character (steer toward Stoic vs flatten |c|)

| item | base c | Seneca c | Δc | Δ|c| | character |
|---|---|---|---|---|---|
| emot_03 | -1.789 | -0.250 | +1.539 | -1.539 | FLATTENS |c| (disrupts discrimination) |
| emot_01 | -1.508 | -0.742 | +0.766 | -0.766 | FLATTENS |c| (disrupts discrimination) |
| ext_04 | -1.766 | -0.883 | +0.883 | -0.883 | FLATTENS |c| (disrupts discrimination) |

Across all 10 items, Seneca reduces |c| on 5 and increases it on 5.
