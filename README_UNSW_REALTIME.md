# UNSW-NB15 runtime guide

Use the current [README](README.md) for setup, model training, replay, packet capture, dashboard and environment variables. The former guide is retained at [docs/original/README-UNSW-original.md](docs/original/README-UNSW-original.md) for provenance.

The default model directory is now `artifacts/deployment`; the former `saved_models` pickles are not automatically loaded. The database defaults to `logs/alerts.db`, not `/home/osboxes/alerts.db`. Run `python train.py --quick` to build a current compatible bundle. Dataset replay and synthetic self-tests are explicitly labelled, and never imply live capture.
