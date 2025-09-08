# storage.py
# =============================================================================
# HRV 时窗特征入库与哈希摘要：DuckDB / SQLite 兼容实现
# =============================================================================
from __future__ import annotations
import os, hashlib, datetime as dt
from typing import Any, Dict, Optional

# 可同时支持 duckdb / sqlite3，优先使用 duckdb（支持 median 等聚合）
_BACKEND = os.environ.get("HRV_DB_BACKEND", "duckdb").lower()

if _BACKEND == "sqlite":
    import sqlite3
    DBLib = sqlite3
else:
    import duckdb
    DBLib = duckdb

# -----------------------------
# 连接与建表
# -----------------------------
def connect(db_path: str):
    """创建数据库连接；DuckDB/SQLite 兼容。"""
    if _BACKEND == "sqlite":
        conn = DBLib.connect(db_path, check_same_thread=False)
    else:
        conn = DBLib.connect(db_path)
    return conn

def init_db(conn) -> None:
    """创建表（若不存在）。使用 feature_key 作为主键。"""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS hrv_features (
        feature_key     TEXT PRIMARY KEY,
        patient_id      TEXT,
        window_start    TIMESTAMP,
        window_end      TIMESTAMP,
        window_length   REAL,
        env_label       TEXT,
        rmssd           REAL,
        sdnn            REAL,
        pnn50           REAL,
        sd1             REAL,
        sd2             REAL,
        sd1_sd2_ratio   REAL,
        lf              REAL,
        hf              REAL,
        lf_hf_ratio     REAL,
        hti             REAL,
        rhythm_status   TEXT,
        pwv             REAL,
        aix             REAL,
        vascular_status TEXT,
        stress_index    REAL,
        stress_level    TEXT,
        anxiety_score   REAL,
        anxiety_level   TEXT,           -- ← 这里务必是 TEXT（字符串）
        sampling_spec   TEXT,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        notes           TEXT
    );
    """)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS feature_hash (
        feature_key    TEXT,
        sha256_digest  TEXT,
        lsh_bucket     INTEGER,
        created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (feature_key, sha256_digest)
    );
    """)




# -----------------------------
# 键与哈希
# -----------------------------
def make_feature_key(patient_id: str, window_start: str, window_end: str) -> str:
    """
    构造确定性 feature_key：SHA256(patient_id|window_start|window_end) 的前 16 字符。
    - 幂等性：同一用户同一窗必然相同
    """
    raw = f"{patient_id}|{window_start}|{window_end}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]

def compute_row_digest(row: Dict[str, Any]) -> str:
    """
    对“按固定字段顺序拼接”的特征行做 SHA-256。
    仅用于完整性校验与隐私保护，不反推原始数据。
    """
    fields = [
        "patient_id","window_start","window_end","window_length","env_label",
        "rmssd","sdnn","pnn50","sd1","sd2","sd1_sd2_ratio",
        "lf","hf","lf_hf_ratio","hti","rhythm_status",
        "pwv","aix","vascular_status",
        "stress_index","stress_level","anxiety_score","anxiety_level"
    ]
    norm = "|".join("" if row.get(k) is None else str(row.get(k)) for k in fields)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()

def lsh_bucket(sha256_digest: str, n_buckets: int = 1024) -> int:
    """把哈希映射到固定桶，便于近似检索；默认 1024 桶。"""
    return int(sha256_digest[:8], 16) % n_buckets

# -----------------------------
# 入库（UPSERT）
# -----------------------------
def upsert_hrv_feature(conn, row: Dict[str, Any]) -> Dict[str, Any]:
    """
    幂等入库：以 feature_key 为主键；存在则 UPDATE，不存在则 INSERT。
    返回 {"id": None, "feature_key": "..."} —— 不再依赖自增 id。
    """
    # 1) 确保 feature_key 存在（patient_id|window_start|window_end 的哈希）
    if not row.get("feature_key"):
        row["feature_key"] = make_feature_key(
            str(row.get("patient_id", "")),
            str(row["window_start"]),
            str(row["window_end"])
        )

    # 2) 列顺序（与 init_db 的表结构一致；不写 created_at 让其走默认值）
    cols = [
        "feature_key",
        "patient_id", "window_start", "window_end", "window_length", "env_label",
        "rmssd", "sdnn", "pnn50", "sd1", "sd2", "sd1_sd2_ratio",
        "lf", "hf", "lf_hf_ratio",
        "hti", "rhythm_status",
        "pwv", "aix", "vascular_status",
        "stress_index", "stress_level", "anxiety_score", "anxiety_level",
        "sampling_spec", "notes"
    ]
    vals = [row.get(c) for c in cols]

    # 3) 判断是否已存在
    existed = conn.execute(
        "SELECT 1 FROM hrv_features WHERE feature_key = ?", [row["feature_key"]]
    ).fetchone()

    if existed:
        # UPDATE：除 feature_key 外的所有列
        set_cols = [c for c in cols if c != "feature_key"]
        set_clause = ", ".join([f"{c} = ?" for c in set_cols])
        params = [row.get(c) for c in set_cols] + [row["feature_key"]]
        conn.execute(
            f"UPDATE hrv_features SET {set_clause} WHERE feature_key = ?",
            params
        )
    else:
        # INSERT：包含 feature_key
        placeholders = ",".join(["?"] * len(cols))
        conn.execute(
            f"INSERT INTO hrv_features ({','.join(cols)}) VALUES ({placeholders})",
            vals
        )

    # 4) 返回：不再依赖 id，自增主键已取消
    return {"id": None, "feature_key": row["feature_key"]}

def insert_feature_hash(conn, feature_key: str, row: Dict[str, Any]) -> None:
    """写入 feature_hash：sha256 摘要 + LSH 桶编号。"""
    digest = compute_row_digest(row)
    bucket = lsh_bucket(digest)
    # 去重写入（同一键+摘要不重复）
    q = "SELECT 1 FROM feature_hash WHERE feature_key = ? AND sha256_digest = ?"
    if not conn.execute(q, [feature_key, digest]).fetchone():
        conn.execute(
            "INSERT INTO feature_hash (feature_key, sha256_digest, lsh_bucket) VALUES (?, ?, ?)",
            [feature_key, digest, bucket]
        )
