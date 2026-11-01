# Data provenance and packaging

The dataset CSVs came from the supplied `IDS-GWO-FeatureSelection(4).zip`. No customer or banking data is used.

- `dataset/raw_unsw/UNSW_NB15_train.csv` and `UNSW_NB15_test.csv`: supplied headered UNSW-NB15 benchmark split.
- `dataset/raw_unsw/NUSW-NB15_features.csv`: supplied feature definitions, including millisecond units.
- `dataset/raw/KDDTrain.csv` and `KDDTest.csv`: supplied NSL-KDD-derived 41-feature, multiclass CSVs.
- `examples/flows.csv`: reproducible 50-row seed-42 sample of supplied UNSW test features, with labels excluded.

Dataset provenance should be credited in the project report. Consult the original dataset provider's research-use and redistribution terms before publishing the package publicly. This handoff returns the user's supplied data; it does not grant a new dataset licence or imply that licensing was legally reviewed.

The original ZIP contained approximately 586 MB of four-part full UNSW raw CSVs plus redundant processed copies. Those are not needed by the default presplit workflow and are excluded from the updated compact ZIP. They remain untouched in the original archive. Full-raw loading code and placement instructions are preserved; copy those four files back and set `IDS_DATA_SOURCE=raw_full` to use that optional mode.

Old uploaded model pickles were not executed. The included deployment model was trained during this work from the supplied CSVs. No local event database, credentials, virtual environment, browser binary or cache is included. Research outputs are explicitly measured reports, not live observations.
