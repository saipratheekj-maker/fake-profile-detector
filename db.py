"""
db.py — MySQL database layer for ProfileSentinel
All queries are parameterised to prevent SQL injection.
"""

import json
import mysql.connector
from mysql.connector import pooling, Error as MySQLError

try:
    import config
    DB_CONFIG = {
        "host":     config.DB_HOST,
        "port":     config.DB_PORT,
        "database": config.DB_NAME,
        "user":     config.DB_USER,
        "password": config.DB_PASSWORD,
        "charset":  "utf8mb4",
    }
except ImportError:
    # Fallback defaults — update config.py with real credentials
    DB_CONFIG = {
        "host":     "localhost",
        "port":     3306,
        "database": "profile_sentinel",
        "user":     "root",
        "password": "",
        "charset":  "utf8mb4",
    }

_pool = None


def get_pool():
    global _pool
    if _pool is None:
        _pool = pooling.MySQLConnectionPool(
            pool_name="sentinel_pool",
            pool_size=5,
            **DB_CONFIG,
        )
    return _pool


def get_conn():
    return get_pool().get_connection()


# ── Write ─────────────────────────────────────────────────────

def save_profile(raw_input: dict, result: dict, features: dict) -> int:
    """
    Persist one analysis to the database.
    Returns the new row's auto-increment id.
    """
    sql = """
        INSERT INTO profiles (
            username, display_name, bio,
            followers_count, following_count, posts_count,
            account_age_days, avg_likes_per_post, avg_comments_per_post,
            has_profile_pic, has_external_url, is_private,
            prediction, fake_probability, real_probability,
            trust_score, confidence, risk_factors, top_features,
            followers_following_ratio, post_frequency,
            username_digit_ratio, bio_length
        ) VALUES (
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s, %s,
            %s, %s,
            %s, %s
        )
    """
    values = (
        raw_input.get("username", ""),
        raw_input.get("display_name", ""),
        raw_input.get("bio", ""),
        raw_input.get("followers_count", 0),
        raw_input.get("following_count", 0),
        raw_input.get("posts_count", 0),
        raw_input.get("account_age_days", 0),
        raw_input.get("avg_likes_per_post", 0),
        raw_input.get("avg_comments_per_post", 0),
        int(raw_input.get("has_profile_pic", False)),
        int(raw_input.get("has_external_url", False)),
        int(raw_input.get("is_private", False)),
        result["prediction"],
        result["fake_probability"],
        result["real_probability"],
        result["trust_score"],
        result["confidence"],
        json.dumps(result.get("risk_factors", [])),
        json.dumps(result.get("top_features", [])),
        features.get("followers_following_ratio", 0),
        features.get("post_frequency", 0),
        features.get("username_digit_ratio", 0),
        features.get("bio_length", 0),
    )

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, values)
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def update_notes(profile_id: int, notes: str) -> bool:
    sql = "UPDATE profiles SET notes = %s WHERE id = %s"
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, (notes, profile_id))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def delete_profile(profile_id: int) -> bool:
    sql = "DELETE FROM profiles WHERE id = %s"
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(sql, (profile_id,))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


# ── Read ──────────────────────────────────────────────────────

def _row_to_dict(cursor, row):
    columns = [d[0] for d in cursor.description]
    d = dict(zip(columns, row))
    # Deserialise JSON columns
    for col in ("risk_factors", "top_features"):
        if col in d and isinstance(d[col], str):
            try:
                d[col] = json.loads(d[col])
            except Exception:
                d[col] = []
    # Convert non-serialisable types
    for k, v in d.items():
        if hasattr(v, "isoformat"):          # datetime → string
            d[k] = v.isoformat(sep=" ")
    return d


def get_profiles(
    page: int = 1,
    per_page: int = 20,
    search: str = "",
    filter_prediction: str = "",
    sort_by: str = "analyzed_at",
    sort_dir: str = "desc",
) -> dict:
    allowed_sort   = {"analyzed_at", "username", "fake_probability", "trust_score", "followers_count"}
    allowed_dir    = {"asc", "desc"}
    sort_by  = sort_by  if sort_by  in allowed_sort else "analyzed_at"
    sort_dir = sort_dir if sort_dir in allowed_dir  else "desc"

    where_clauses = []
    params        = []

    if search:
        where_clauses.append("(username LIKE %s OR display_name LIKE %s)")
        params += [f"%{search}%", f"%{search}%"]
    if filter_prediction in ("fake", "real"):
        where_clauses.append("prediction = %s")
        params.append(filter_prediction)

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    offset = (page - 1) * per_page

    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM profiles {where}", params)
        total = cur.fetchone()[0]

        cur.execute(
            f"""
            SELECT id, username, display_name, followers_count, following_count,
                   posts_count, prediction, fake_probability, trust_score,
                   confidence, analyzed_at, notes
            FROM profiles {where}
            ORDER BY {sort_by} {sort_dir}
            LIMIT %s OFFSET %s
            """,
            params + [per_page, offset],
        )
        rows = [_row_to_dict(cur, r) for r in cur.fetchall()]
        return {
            "profiles": rows,
            "total":    total,
            "page":     page,
            "per_page": per_page,
            "pages":    max(1, -(-total // per_page)),  # ceiling div
        }
    finally:
        conn.close()


def get_profile_by_id(profile_id: int) -> dict | None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM profiles WHERE id = %s", (profile_id,))
        row = cur.fetchone()
        return _row_to_dict(cur, row) if row else None
    finally:
        conn.close()


def get_stats() -> dict:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM vw_stats")
        row = cur.fetchone()
        if not row:
            return {}
        stats = _row_to_dict(cur, row)

        # Trend: last 7 days counts
        cur.execute("""
            SELECT DATE(analyzed_at) AS day,
                   COUNT(*) AS total,
                   SUM(prediction='fake') AS fake
            FROM profiles
            WHERE analyzed_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
            GROUP BY day
            ORDER BY day
        """)
        stats["daily_trend"] = [_row_to_dict(cur, r) for r in cur.fetchall()]

        return stats
    finally:
        conn.close()


def test_connection() -> bool:
    try:
        conn = get_conn()
        conn.close()
        return True
    except MySQLError:
        return False
