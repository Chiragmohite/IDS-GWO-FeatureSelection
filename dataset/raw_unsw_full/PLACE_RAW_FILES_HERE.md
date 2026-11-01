# Place your raw UNSW-NB15 partition files here

This project's default configuration (`config.UNSW_DATA_SOURCE = "raw_full"`)
expects these 4 files in this folder:

```
dataset/raw_unsw_full/UNSW-NB15_1.csv
dataset/raw_unsw_full/UNSW-NB15_2.csv
dataset/raw_unsw_full/UNSW-NB15_3.csv
dataset/raw_unsw_full/UNSW-NB15_4.csv
```

They were left out of this project zip because, combined, they're ~560MB
-- too large to ship inside the project archive. You already have them
(from the original UNSW-NB15 download / the zip you extracted them
from). Just copy all 4 CSV files into this folder and `python train.py`
will work immediately -- `src/preprocessing/unsw_raw_loader.py` handles
everything else (header-less parsing, column renaming, schema
reconciliation, the derived `rate` feature, and a fresh stratified
train/test split).

`UNSW-NB15_LIST_EVENTS.csv` (already included) is just a summary
reference table of attack subcategory counts -- documentation only, not
used by any code in this project.

## Don't have the raw files handy right now?

Switch to the smaller, faster pre-split benchmark option instead: set
`UNSW_DATA_SOURCE = "presplit"` in `config.py`. Those files are already
included in `dataset/raw_unsw/` and require no setup. See
`README_UNSW_REALTIME.md` -> "Dataset options" for the full comparison.
