"""
Fake Profile Detector — Flask API with MySQL
Run:  python app.py  →  open http://127.0.0.1:5000
"""

import os, sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(__file__))
from model.fake_detector import load_model, predict_profile
import db

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = None

def get_model():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model

def build_features(data):
    username         = str(data.get("username", ""))
    display_name     = str(data.get("display_name", ""))
    followers        = int(data.get("followers_count", 0))
    following        = int(data.get("following_count", 0))
    posts            = int(data.get("posts_count", 0))
    has_pic          = int(bool(data.get("has_profile_pic", False)))
    bio              = str(data.get("bio", ""))
    account_age_days = int(data.get("account_age_days", 0))
    avg_likes        = float(data.get("avg_likes_per_post", 0))
    avg_comments     = float(data.get("avg_comments_per_post", 0))
    has_url          = int(bool(data.get("has_external_url", False)))
    is_private       = int(bool(data.get("is_private", False)))

    digit_chars           = sum(c.isdigit() for c in username)
    username_digit_ratio  = digit_chars / max(len(username), 1)
    special_chars         = sum(not c.isalnum() and c not in ("_", ".") for c in username)
    special_char_ratio    = special_chars / max(len(username), 1)
    name_tokens           = display_name.lower().split()
    name_matches_username = int(any(t in username.lower() for t in name_tokens if len(t) > 2))
    ff_ratio              = followers / max(following, 1)
    post_frequency        = (posts / max(account_age_days, 1)) * 30

    raw_input = {
        "username": username, "display_name": display_name, "bio": bio,
        "followers_count": followers, "following_count": following,
        "posts_count": posts, "account_age_days": account_age_days,
        "avg_likes_per_post": avg_likes, "avg_comments_per_post": avg_comments,
        "has_profile_pic": has_pic, "has_external_url": has_url, "is_private": is_private,
    }
    profile_features = {
        "followers_count": followers, "following_count": following,
        "posts_count": posts, "has_profile_pic": has_pic,
        "bio_length": len(bio), "username_digit_ratio": username_digit_ratio,
        "username_length": len(username), "account_age_days": account_age_days,
        "avg_likes_per_post": avg_likes, "avg_comments_per_post": avg_comments,
        "has_external_url": has_url, "is_private": is_private,
        "followers_following_ratio": ff_ratio, "post_frequency": post_frequency,
        "name_matches_username": name_matches_username,
        "special_char_in_username": special_char_ratio,
    }
    return raw_input, profile_features

# ── Static ────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

# ── Detect & auto-save ────────────────────────────────────────
@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    try:
        raw_input, profile_features = build_features(data)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 422

    result = predict_profile(get_model(), profile_features)
    result["trust_score"] = round(result["real_probability"])

    saved_id = None
    db_error = None
    try:
        saved_id = db.save_profile(raw_input, result, profile_features)
    except Exception as e:
        db_error = str(e)

    return jsonify({**result, "profile_features": profile_features,
                    "username": raw_input["username"],
                    "saved_id": saved_id, "db_error": db_error})

# ── History ───────────────────────────────────────────────────
@app.route("/api/profiles", methods=["GET"])
def list_profiles():
    try:
        return jsonify(db.get_profiles(
            page=int(request.args.get("page", 1)),
            per_page=int(request.args.get("per_page", 20)),
            search=request.args.get("search", ""),
            filter_prediction=request.args.get("prediction", ""),
            sort_by=request.args.get("sort_by", "analyzed_at"),
            sort_dir=request.args.get("sort_dir", "desc"),
        ))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/profiles/<int:pid>", methods=["GET"])
def get_profile(pid):
    row = db.get_profile_by_id(pid)
    return jsonify(row) if row else (jsonify({"error": "Not found"}), 404)

@app.route("/api/profiles/<int:pid>", methods=["DELETE"])
def delete_profile(pid):
    return jsonify({"deleted": db.delete_profile(pid), "id": pid})

@app.route("/api/profiles/<int:pid>/notes", methods=["PATCH"])
def update_notes(pid):
    body = request.get_json(force=True) or {}
    ok   = db.update_notes(pid, str(body.get("notes", "")))
    return jsonify({"updated": ok})

# ── Stats & system ────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def stats():
    try:
        return jsonify(db.get_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({
        "status":       "ok",
        "model_loaded": get_model() is not None,
        "db_connected": db.test_connection(),
    })

@app.route("/api/demo", methods=["GET"])
def demo_profiles():
    return jsonify([
        {"label": "Typical Real User", "username": "jessica_travels",
         "display_name": "Jessica M.", "followers_count": 1240, "following_count": 890,
         "posts_count": 134, "has_profile_pic": True,
         "bio": "Travel photographer | Coffee addict | Based in NYC",
         "account_age_days": 1200, "avg_likes_per_post": 87,
         "avg_comments_per_post": 5, "has_external_url": True, "is_private": False},
        {"label": "Obvious Fake Bot", "username": "user4829301x",
         "display_name": "User", "followers_count": 12, "following_count": 6800,
         "posts_count": 2, "has_profile_pic": False, "bio": "",
         "account_age_days": 7, "avg_likes_per_post": 0,
         "avg_comments_per_post": 0, "has_external_url": False, "is_private": False},
        {"label": "Inflated Follower Account", "username": "influencer_king99",
         "display_name": "King", "followers_count": 95000, "following_count": 450,
         "posts_count": 18, "has_profile_pic": True, "bio": "DM for collabs",
         "account_age_days": 45, "avg_likes_per_post": 3,
         "avg_comments_per_post": 0.1, "has_external_url": False, "is_private": False},
    ])

if __name__ == "__main__":
    print("[*] Pre-loading model...")
    get_model()
    print("[*] Testing DB connection...")
    if db.test_connection():
        print("[+] MySQL connected OK")
    else:
        print("[!] MySQL unavailable — edit config.py with your credentials")
    print("[+] Starting server on http://127.0.0.1:5001")
    app.run(debug=True, port=5001)
