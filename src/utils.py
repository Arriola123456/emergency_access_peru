"""Utilities compartidos: normalización de UBIGEO, logging, helpers varios."""

from __future__ import annotations

import logging
import re
from typing import Iterable

import pandas as pd


def get_logger(name: str) -> logging.Logger:
    """Logger con formato consistente; idempotente si ya fue configurado."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s — %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


_UBIGEO_RE = re.compile(r"\D")


def normalize_ubigeo(series: pd.Series, length: int = 6) -> pd.Series:
    """Normaliza códigos UBIGEO: strip non-digits, zero-pad izquierda, trunca/rellena a length."""
    cleaned = series.astype("string").str.replace(_UBIGEO_RE, "", regex=True)
    cleaned = cleaned.where(cleaned.str.len() > 0, other=pd.NA)
    cleaned = cleaned.str.zfill(length).str.slice(0, length)
    return cleaned


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Devuelve el primer nombre de columna de `candidates` que exista en `df`, o None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
