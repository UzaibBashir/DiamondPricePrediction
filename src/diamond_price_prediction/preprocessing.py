"""Data loading, cleaning, and feature engineering for diamond prices."""

import pandas as pd
import numpy as np


# Canonical ordinal orderings for the categorical features
CUT_ORDER     = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
COLOR_ORDER   = ["J", "I", "H", "G", "F", "E", "D"]      # D = best, J = worst -> we encode worst→best
CLARITY_ORDER = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(filepath="data/diamonds.csv"):
    """Load the ggplot2 diamonds dataset (53,940 rows)."""
    return pd.read_csv(filepath)


# ──────────────────────────────────────────────
# Data Cleaning
# ──────────────────────────────────────────────

def drop_zero_dimensions(df):
    """A small fraction of rows have x/y/z == 0 — physically impossible. Drop them."""
    df_clean = df.copy()
    bad = (df_clean["x"] == 0) | (df_clean["y"] == 0) | (df_clean["z"] == 0)
    df_clean = df_clean[~bad].reset_index(drop=True)
    return df_clean


def encode_ordinals(df):
    """Encode cut, color, clarity as integers in the canonical worst→best order."""
    df_clean = df.copy()
    df_clean["cut"] = df_clean["cut"].map({k: i for i, k in enumerate(CUT_ORDER)})
    df_clean["color"] = df_clean["color"].map({k: i for i, k in enumerate(COLOR_ORDER)})
    df_clean["clarity"] = df_clean["clarity"].map({k: i for i, k in enumerate(CLARITY_ORDER)})
    return df_clean


# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────

def create_features(df):
    """
    Engineer:
      - volume       = x * y * z
      - density      = carat / volume   (carat is mass, volume from x,y,z)
      - price_log    (only used in EDA — not on the cleaned set)
      - log_carat
    """
    df_feat = df.copy()
    df_feat["volume"] = df_feat["x"] * df_feat["y"] * df_feat["z"]
    df_feat["density"] = df_feat["carat"] / df_feat["volume"].replace(0, np.nan)
    df_feat["log_carat"] = np.log1p(df_feat["carat"])
    return df_feat


def preprocess_data(df):
    """Full pipeline: drop zero-dim rows, ordinal-encode, engineer features."""
    df_clean = drop_zero_dimensions(df)
    df_clean = encode_ordinals(df_clean)
    df_clean = create_features(df_clean)
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))
    return df_clean


