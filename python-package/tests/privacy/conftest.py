"""Shared fixtures for the group's privacy test suite (Privacy and Big Data, KU Leuven).

These tests describe the privacy behaviour we expect from the access control
deployment described in our PIA. They are written to FAIL until the matching
recommendation is implemented, so that the development team can see which
privacy issues are still open.
"""
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from insightface.gui.core.storage import Storage

EMBEDDING_DIM = 512  # ArcFace w600k_mbf in the cheetah_s package


@pytest.fixture
def gallery(tmp_path):
    """A fresh SQLite gallery, created exactly as the Evaluation Studio does."""
    db_path = tmp_path / "insightface_gui.db"
    return Storage(db_path), db_path


@pytest.fixture
def employee_embedding():
    """A deterministic, L2-normalised 512-d face template."""
    vector = np.random.default_rng(seed=2026).normal(size=EMBEDDING_DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture
def face_crop(tmp_path):
    """A face crop file on disk, as written to the workspace ``crops/`` folder."""
    crops = tmp_path / "crops"
    crops.mkdir()
    path = crops / "employee_001.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0 fake JPEG face crop")
    return path


def _raw_database_bytes(db_path: Path) -> bytes:
    data = Path(db_path).read_bytes()
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(db_path) + suffix)
        if sidecar.exists():
            data += sidecar.read_bytes()
    return data


def _query_one(db_path: Path, sql: str, params=()):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql, params).fetchone()[0]


@pytest.fixture
def raw_bytes():
    """Everything an adversary with file access to zone Z4 can read."""
    return _raw_database_bytes


@pytest.fixture
def query_one():
    """Run a scalar SQL query directly on the database file."""
    return _query_one
