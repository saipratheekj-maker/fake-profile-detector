# 🛡️ ProfileSentinel — Fake Profile Detector

Detect fake social media profiles using a trained **Random Forest** machine learning model.
Built with Python (scikit-learn + Flask) for the backend and HTML/CSS/JS for the frontend.

---

## 🏗️ Project Structure

```
fake-profile-detector/
├── app.py                   # Flask web server & REST API
├── requirements.txt         # Python dependencies
├── model.pkl                # Auto-generated trained model (created on first run)
├── model/
│   └── fake_detector.py     # ML model: data generation, training, prediction
└── static/
    └── index.html           # Frontend (HTML + CSS + JS in one file)
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python app.py
```

The server will:
1. Auto-train the ML model on **8,000 synthetic profiles** (first run only, ~5 seconds)
2. Save the model to `model.pkl`
3. Start Flask on **http://127.0.0.1:5001**

### 3. Open the app
Visit **http://127.0.0.1:5001** in your browser.

---

## 🤖 ML Model Details

### Algorithm
- **Random Forest Classifier** (200 estimators, max_depth=12)
- Preprocessing via **StandardScaler** (in a sklearn Pipeline)
- Balanced class weights to handle imbalanced data

### Features (16 total)
| Feature | Description |
|---|---|
| `followers_count` | Number of followers |
| `following_count` | Number of accounts followed |
| `posts_count` | Total posts made |
| `has_profile_pic` | 0 or 1 |
| `bio_length` | Character length of bio |
| `username_digit_ratio` | Proportion of digits in username |
| `username_length` | Length of username string |
| `account_age_days` | Days since account creation |
| `avg_likes_per_post` | Average likes across posts |
| `avg_comments_per_post` | Average comments across posts |
| `has_external_url` | 0 or 1 |
| `is_private` | 0 or 1 |
| `followers_following_ratio` | followers / following |
| `post_frequency` | Posts per month |
| `name_matches_username` | 1 if display name appears in username |
| `special_char_in_username` | Ratio of special characters |

### Synthetic Training Data
The model trains on **8,000 synthetic profiles** designed to mimic real behavioural patterns:
- **Real profiles**: organic follower/following ratios, profile pictures, meaningful bios, normal engagement
- **Fake profiles**: spam following patterns, no photos, digit-heavy usernames, new accounts, near-zero engagement

---

## 🌐 API Endpoints

### `POST /api/detect`
Analyze a profile.

**Request body:**
```json
{
  "username": "user123",
  "display_name": "John",
  "bio": "Just a regular user",
  "followers_count": 120,
  "following_count": 4500,
  "posts_count": 3,
  "account_age_days": 14,
  "avg_likes_per_post": 0.5,
  "avg_comments_per_post": 0,
  "has_profile_pic": false,
  "has_external_url": false,
  "is_private": false
}
```

**Response:**
```json
{
  "prediction": "fake",
  "fake_probability": 94.3,
  "real_probability": 5.7,
  "confidence": "high",
  "risk_factors": ["Very new account (< 30 days)", "Very low engagement"],
  "top_features": [["followers_following_ratio", 0.18], ...],
  "trust_score": 6
}
```

### `GET /api/demo`
Returns 3 pre-built demo profiles.

### `GET /api/health`
Returns model status.

---

## 🔧 Extending the Model

To retrain with different parameters, edit `model/fake_detector.py`:
```python
# Adjust RandomForestClassifier params
"clf": RandomForestClassifier(
    n_estimators=200,   # More trees = more accurate, slower
    max_depth=12,       # Deeper = more complex patterns
    ...
)
```

To use real training data, replace `generate_synthetic_data()` with a CSV loader.

---

## 📦 Tech Stack
- **Backend**: Python 3.10+, Flask, scikit-learn, NumPy
- **Frontend**: Vanilla HTML/CSS/JS (no framework dependencies)
- **Font**: Syne + Space Mono (Google Fonts)
