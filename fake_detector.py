import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline


FEATURE_NAMES = [
    "followers_count",
    "following_count",
    "posts_count",
    "has_profile_pic",
    "bio_length",
    "username_digit_ratio",
    "username_length",
    "account_age_days",
    "avg_likes_per_post",
    "avg_comments_per_post",
    "has_external_url",
    "is_private",
    "followers_following_ratio",
    "post_frequency",           # posts per month
    "name_matches_username",    # 0 or 1
    "special_char_in_username", # ratio
]


def generate_synthetic_data(n_samples=5000, random_state=42):
    """
    Generate synthetic training data that mimics real/fake profile patterns.
    Real profiles tend to have:
      - More varied follower/following ratios
      - Profile pictures
      - Non-zero bios
      - Organic engagement ratios
    Fake profiles tend to have:
      - Extreme follower/following ratios or very low counts
      - No profile picture
      - Short/empty bios
      - Digit-heavy usernames
      - Very new accounts
    """
    rng = np.random.default_rng(random_state)
    n_real = n_samples // 2
    n_fake = n_samples - n_real

    # --- Real profiles ---
    real_followers  = rng.integers(50, 5000, n_real)
    real_following  = rng.integers(50, 2000, n_real)
    real_posts      = rng.integers(5,  500,  n_real)
    real_pic        = rng.choice([0, 1], n_real, p=[0.05, 0.95])
    real_bio_len    = rng.integers(10, 200, n_real)
    real_dig_ratio  = rng.uniform(0, 0.15, n_real)
    real_usr_len    = rng.integers(5, 20, n_real)
    real_age        = rng.integers(180, 3650, n_real)
    real_avg_likes  = rng.uniform(5, 300, n_real)
    real_avg_cmts   = rng.uniform(0.5, 30, n_real)
    real_url        = rng.choice([0, 1], n_real, p=[0.6, 0.4])
    real_private    = rng.choice([0, 1], n_real, p=[0.7, 0.3])
    real_ff_ratio   = real_followers / (real_following + 1)
    real_post_freq  = rng.uniform(0.5, 15, n_real)
    real_name_match = rng.choice([0, 1], n_real, p=[0.4, 0.6])
    real_spec_char  = rng.uniform(0, 0.1, n_real)

    # --- Fake profiles ---
    fake_followers  = rng.integers(0, 300, n_fake)
    fake_following  = rng.integers(500, 7500, n_fake)
    fake_posts      = rng.integers(0, 20, n_fake)
    fake_pic        = rng.choice([0, 1], n_fake, p=[0.55, 0.45])
    fake_bio_len    = rng.integers(0, 50, n_fake)
    fake_dig_ratio  = rng.uniform(0.2, 0.8, n_fake)
    fake_usr_len    = rng.integers(8, 30, n_fake)
    fake_age        = rng.integers(1, 180, n_fake)
    fake_avg_likes  = rng.uniform(0, 5, n_fake)
    fake_avg_cmts   = rng.uniform(0, 1, n_fake)
    fake_url        = rng.choice([0, 1], n_fake, p=[0.85, 0.15])
    fake_private    = rng.choice([0, 1], n_fake, p=[0.5, 0.5])
    fake_ff_ratio   = fake_followers / (fake_following + 1)
    fake_post_freq  = rng.uniform(0, 2, n_fake)
    fake_name_match = rng.choice([0, 1], n_fake, p=[0.8, 0.2])
    fake_spec_char  = rng.uniform(0.1, 0.5, n_fake)

    real_X = np.column_stack([
        real_followers, real_following, real_posts, real_pic, real_bio_len,
        real_dig_ratio, real_usr_len, real_age, real_avg_likes, real_avg_cmts,
        real_url, real_private, real_ff_ratio, real_post_freq,
        real_name_match, real_spec_char
    ])
    fake_X = np.column_stack([
        fake_followers, fake_following, fake_posts, fake_pic, fake_bio_len,
        fake_dig_ratio, fake_usr_len, fake_age, fake_avg_likes, fake_avg_cmts,
        fake_url, fake_private, fake_ff_ratio, fake_post_freq,
        fake_name_match, fake_spec_char
    ])

    X = np.vstack([real_X, fake_X])
    y = np.hstack([np.zeros(n_real), np.ones(n_fake)])  # 0 = real, 1 = fake

    # Add some noise to make it realistic
    noise = rng.normal(0, 0.01, X.shape)
    X = X + noise
    X = np.clip(X, 0, None)

    return X, y


def train_model(save_path="model.pkl"):
    print("[*] Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples=8000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("[*] Training Random Forest model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[+] Model accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"[+] Model saved to {save_path}")
    return pipeline


def load_model(model_path="model.pkl"):
    if not os.path.exists(model_path):
        print("[!] Model not found, training now...")
        return train_model(model_path)
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict_profile(model, profile_features: dict):
    """
    profile_features keys (all numeric):
      followers_count, following_count, posts_count, has_profile_pic,
      bio_length, username_digit_ratio, username_length, account_age_days,
      avg_likes_per_post, avg_comments_per_post, has_external_url, is_private,
      followers_following_ratio, post_frequency, name_matches_username,
      special_char_in_username
    """
    feature_vector = np.array([[
        profile_features.get("followers_count", 0),
        profile_features.get("following_count", 0),
        profile_features.get("posts_count", 0),
        profile_features.get("has_profile_pic", 0),
        profile_features.get("bio_length", 0),
        profile_features.get("username_digit_ratio", 0),
        profile_features.get("username_length", 0),
        profile_features.get("account_age_days", 0),
        profile_features.get("avg_likes_per_post", 0),
        profile_features.get("avg_comments_per_post", 0),
        profile_features.get("has_external_url", 0),
        profile_features.get("is_private", 0),
        profile_features.get("followers_following_ratio", 0),
        profile_features.get("post_frequency", 0),
        profile_features.get("name_matches_username", 0),
        profile_features.get("special_char_in_username", 0),
    ]])

    prediction = model.predict(feature_vector)[0]
    proba = model.predict_proba(feature_vector)[0]
    fake_probability = float(proba[1])

    # Feature importances for explanation
    rf = model.named_steps["clf"]
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:5]
    top_features = [(FEATURE_NAMES[i], round(float(importances[i]), 4)) for i in top_indices]

    risk_factors = []
    ff_ratio = profile_features.get("followers_following_ratio", 1)
    if ff_ratio < 0.1:
        risk_factors.append("Very low followers-to-following ratio")
    if profile_features.get("has_profile_pic", 1) == 0:
        risk_factors.append("No profile picture")
    if profile_features.get("bio_length", 100) < 10:
        risk_factors.append("Empty or very short bio")
    if profile_features.get("username_digit_ratio", 0) > 0.3:
        risk_factors.append("Username contains many digits")
    if profile_features.get("account_age_days", 365) < 30:
        risk_factors.append("Very new account (< 30 days)")
    if profile_features.get("posts_count", 10) < 3:
        risk_factors.append("Very few posts")
    if profile_features.get("avg_likes_per_post", 10) < 1:
        risk_factors.append("Unusually low engagement")
    if profile_features.get("following_count", 0) > 3000 and profile_features.get("followers_count", 0) < 100:
        risk_factors.append("Following many but has few followers")

    return {
        "prediction": "fake" if prediction == 1 else "real",
        "fake_probability": round(fake_probability * 100, 2),
        "real_probability": round((1 - fake_probability) * 100, 2),
        "confidence": "high" if abs(fake_probability - 0.5) > 0.35 else "medium" if abs(fake_probability - 0.5) > 0.15 else "low",
        "risk_factors": risk_factors,
        "top_features": top_features,
    }


if __name__ == "__main__":
    train_model("model.pkl")
