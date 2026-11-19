"""Data access and preprocessing for the NASA C-MAPSS dataset.

Modules
-------
load_data   reading raw files, user uploads and downloading the NASA archive
            (``python -m src.data.load_data --download``)
preprocess  RUL targets, engine-level splits, truncation sampling, CV splitter
"""
