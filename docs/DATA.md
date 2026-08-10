# Data provenance

No third-party data is redistributed in this repository. Each source below must be
obtained from its original provider under that provider's licence.

| Source | Role in the paper | How to obtain | Licence |
|---|---|---|---|
| NHANES 2013-2014 (demographic, examination, questionnaire) | L1 vs L2 on a nationally representative sample; the only source with survey design variables | CDC / NCHS | US Government public domain |
| LifeSnaps | Backbone for L3 and L4: 71 adults, median 88 days each, behaviour, sleep, heart rate, HRV, momentary self-report, step goal | Published dataset, see the paper's citation | Per the dataset's own terms |
| A second public Fitbit cohort | Independent replication: 33 people, 31 days, no demographics | Public dataset | Per the dataset's own terms |

## Datasets screened and rejected

Five candidate datasets were excluded as fabricated. The evidence is in the paper's
appendix and in `outputs/texts/e2_battery_results.csv`. They are named there so the
adjudication can be checked. "Fabricated" denotes a structural signature in the file
(a lookup table, independently drawn columns, rows that do not persist through time,
or derived quantities contradicting their inputs), not an established fact about any
uploader's intent.
