# 🎬 Movie Recommendation System

A free AI-powered movie recommendation system built with Django. Type any movie name and instantly get similar movie suggestions — no signup required.

**🌐 Live Demo:** [movie-recommendation-site.onrender.com](https://movie-recommendation-site-xn4n.onrender.com/)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 AI Recommendations | Content-based cosine similarity across 26,000+ movies |
| 🌐 TMDB Live Fallback | New & regional movies fetched live from TMDB API |
| 🗣️ Same-language first | Recommends movies in the same language first |
| 🎭 Cast/Director Match | If no same-language similar movies found, finds films by the same actors/director |
| 🌍 Other Languages | Second section shows similar movies from other languages |
| 🎬 Genre Browse | Live TMDB data — filter by language, decade, sort by rating/popularity/year |
| 🔥 Trending | 4 tabs: Popular, Now Playing, Top Rated, Upcoming — live from TMDB |
| 😊 Mood Discovery | Find movies by mood (Feel-Good, Thrilling, Scary, etc.) — ~60 movies per mood |
| 🎲 Surprise Me | Random well-rated movie from TMDB, redirects to full detail page |
| 🎞️ Movie Detail Page | Backdrop, poster, tagline, runtime, watch providers (Netflix, Prime, etc.) |
| 👤 User Accounts | Register, login, profiles, edit bio |
| 📋 Watchlist | Save movies to watch later |
| ⭐ Ratings & Reviews | Rate 1–10, write reviews, like & comment |
| 📝 Movie Lists | Create and share public/private movie lists |
| 🏆 Leaderboard | Top reviewers and most-reviewed movies |
| 🌙 Dark/Light Mode | Theme toggle persisted in localStorage |
| 📱 Fully Responsive | Works on mobile and desktop |

---

## 🖼️ How It Works

```
User searches "Amaran"
        ↓
Local dataset (26,326 movies) — not found
        ↓
TMDB API: search by title → get movie ID
        ↓
TMDB /similar → filter by same language (Tamil)
        ↓
If empty → TMDB Credits API (cast + director)
        → Discover movies with same cast/director in Tamil
        ↓
Same-language results shown first
Other-language results shown below
```

For movies **in the local dataset** (e.g. "Inception"):
```
Local top-20 similarity neighbors
        ↓
Split by original_language
        ↓
Same-language section → Other-language section
If same-lang empty → TMDB cast/director/language discovery
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 5.2 (Python 3.11) |
| ML / Similarity | NumPy, SciPy, scikit-learn (TF-IDF + SVD + cosine similarity) |
| Data | Pandas, PyArrow (Parquet) |
| Live Movie Data | TMDB API v3 |
| Frontend | Vanilla JS + CSS (no framework) |
| Auth | Django built-in auth |
| Static Files | WhiteNoise + Brotli compression |
| Database | SQLite |
| Deployment | Render (free tier) |

---

## 📁 Project Structure

```
Movie-Recommendation-System/
├── manage.py
├── requirements.txt
├── build.sh                        # Render build script
├── convert_to_topk.py              # Compress full similarity matrix → top-K
├── models/
│   ├── movie_metadata.parquet      # 26,326 movies metadata
│   ├── top_k_similarities.npz     # Top-20 neighbors per movie (~3 MB)
│   ├── svd_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── title_to_idx.json
│   └── config.json
├── movie_recommendation/
│   ├── settings.py
│   └── urls.py
└── recommender/
    ├── views.py                    # All views + TMDB live logic
    ├── models.py                   # Rating, Watchlist, Review, MovieList
    ├── urls.py
    ├── templatetags/
    │   └── movie_filters.py        # Custom template filter (split)
    └── templates/recommender/
        ├── base.html               # Nav, footer, dark/light mode
        ├── index.html              # Home page
        ├── result.html             # Recommendation results
        ├── tmdb_movie_detail.html  # TMDB movie detail page
        ├── genre_browse.html       # Genre browsing with filters
        ├── trending.html           # Trending tabs
        ├── mood.html / mood_results.html
        └── ...
```

---

## 🚀 Local Setup

```bash
# Clone
git clone https://github.com/srini0406/movie-recommendation-site.git
cd movie-recommendation-site

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Create .env file
SECRET_KEY=your-secret-key-here
DEBUG=True
TMDB_API_KEY=your-tmdb-api-key

# Run migrations
python manage.py migrate

# Start server
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000)

> **Note:** Model files in the `models/` directory are required. `top_k_similarities.npz` is tracked via Git LFS.

---

## ⚙️ Environment Variables

| Variable | Description | Required |
|---|---|---|
| `SECRET_KEY` | Django secret key | ✅ |
| `DEBUG` | `True` for dev, `False` for prod | ✅ |
| `TMDB_API_KEY` | API key from [themoviedb.org](https://www.themoviedb.org/settings/api) | ✅ |
| `MODEL_DIR` | Path to models directory (default: `models`) | ❌ |

---

## 📦 Similarity Matrix Compression

The original full NxN cosine similarity matrix is ~3.5 GB. The `convert_to_topk.py` script compresses it to only store the top-20 neighbors per movie:

```bash
python convert_to_topk.py
```

| File | Size | Notes |
|---|---|---|
| `similarity_matrix.npz` | ~3,500 MB | Original full matrix (not in repo) |
| `top_k_similarities.npz` | ~3 MB | Top-20 neighbors only |
| Reduction | **99.9%** | |

---

## ☁️ Deploying to Render

1. Push to GitHub
2. New Web Service → connect repo
3. Build command: `chmod +x build.sh && bash build.sh`
4. Start command: `gunicorn movie_recommendation.wsgi:application --workers 2 --timeout 120`
5. Environment variables:

| Key | Value |
|---|---|
| `SECRET_KEY` | generate a strong random key |
| `DEBUG` | `False` |
| `TMDB_API_KEY` | your TMDB key |
| `MODEL_DIR` | `models` |
| `PYTHON_VERSION` | `3.11.0` |

---

## 👨‍💻 Developer

**Srinivasan V**
PSG College of Technology
📧 [srinivasanv0401@gmail.com](mailto:srinivasanv0401@gmail.com)

---

## 📄 License

MIT License — free to use, modify and distribute.
