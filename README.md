# 🎬 Movie Recommendation System

A free AI-powered movie recommendation system built with Django. Type any movie name and instantly get similar movie suggestions based on content similarity — no signup required.

**Live Demo:** [movie-recommendation-site.onrender.com](https://movie-recommendation-site.onrender.com)

---

## Features

- **AI Recommendations** — Content-based filtering using cosine similarity across 26,000+ movies
- **TMDB Live Fallback** — New & Tamil movies not in the local dataset are fetched live from TMDB API
- **Same-language first** — Recommendations prioritize the same language as the searched movie
- **Genre Browse** — Live TMDB data, filter by language, decade, sort by rating/popularity/year
- **Trending & Top Rated** — Popularity-ranked movies with poster cards
- **Mood-based Discovery** — Find movies by mood (Feel-Good, Thrilling, Scary, etc.)
- **Surprise Me** — Random well-rated movie picker
- **User Accounts** — Register, login, profiles
- **Watchlist** — Save movies to watch later
- **Ratings & Reviews** — Rate movies 1–10, write reviews, like & comment
- **Movie Lists** — Create and share public/private movie lists
- **Leaderboard** — Top reviewers and most-reviewed movies
- **Dark/Light Mode** — Theme toggle
- **Fully Responsive** — Works on mobile and desktop

---

## Screenshots

> Search any movie → get instant similar movie recommendations with posters, ratings, and links.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Django 5.2 (Python 3.11) |
| ML / Similarity | NumPy, SciPy, scikit-learn |
| Data | Pandas, PyArrow (Parquet) |
| Live Movie Data | TMDB API |
| Frontend | Vanilla JS, CSS (no framework) |
| Auth | Django built-in auth |
| Static Files | WhiteNoise + Brotli |
| Database | SQLite (dev) |
| Deployment | Render |

---

## How It Works

1. **Offline phase** — A TF-IDF vectorizer + SVD model computes cosine similarity across movie metadata (genres, overview, cast, keywords). The full NxN similarity matrix (~3.5 GB) is compressed to a top-20 neighbors format (`top_k_similarities.npz`, ~3 MB).

2. **At request time** — User searches a movie → local dataset lookup → if found, returns top-20 similar movies split by language → if not found, TMDB API is called live to fetch similar movies.

```
User searches "Amaran"
  ↓
Local dataset (26,326 movies) — not found
  ↓
TMDB API live search → fetch similar movies
  ↓
Results returned with posters, ratings, genres
```

---

## Project Structure

```
Movie-Recommendation-System/
├── manage.py
├── requirements.txt
├── build.sh                  # Render build script
├── convert_to_topk.py        # One-time: compress full similarity matrix to top-K
├── models/
│   ├── movie_metadata.parquet
│   ├── top_k_similarities.npz  # shape (26326, 20)
│   ├── svd_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── title_to_idx.json
│   └── config.json
├── movie_recommendation/
│   ├── settings.py
│   └── urls.py
└── recommender/
    ├── views.py              # All views + TMDB fallback logic
    ├── models.py             # Rating, Watchlist, Review, MovieList, etc.
    ├── urls.py
    └── templates/recommender/
        ├── base.html
        ├── index.html
        ├── result.html
        ├── movie_detail.html
        ├── genre_browse.html
        ├── trending.html
        ├── mood.html
        └── ...
```

---

## Local Setup

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

# Set environment variables (create .env file)
echo SECRET_KEY=your-secret-key > .env
echo DEBUG=True >> .env
echo TMDB_API_KEY=your-tmdb-api-key >> .env

# Run migrations
python manage.py migrate

# Start server
python manage.py runserver
```

Open [http://localhost:8000](http://localhost:8000)

> **Note:** You need the model files (`models/` directory) to run locally. The `top_k_similarities.npz` file is included in the repo via Git LFS.

---

## Environment Variables

| Variable | Description | Required |
|---|---|---|
| `SECRET_KEY` | Django secret key | Yes |
| `DEBUG` | `True` for dev, `False` for prod | Yes |
| `TMDB_API_KEY` | API key from [themoviedb.org](https://www.themoviedb.org/settings/api) | Yes (for live data) |
| `MODEL_DIR` | Path to models directory (default: `models`) | No |

---

## Similarity Matrix Compression

The original full cosine similarity matrix is ~3.5 GB. The `convert_to_topk.py` script compresses it to only store the top-20 neighbors per movie:

```bash
python convert_to_topk.py
```

| File | Size |
|---|---|
| `similarity_matrix.npz` (original) | ~3,500 MB |
| `top_k_similarities.npz` (compressed) | ~3 MB |
| Reduction | **99.9%** |

---

## Deployment (Render)

1. Push to GitHub
2. Create a new Web Service on [render.com](https://render.com)
3. Set build command: `chmod +x build.sh && bash build.sh`
4. Set start command: `gunicorn movie_recommendation.wsgi:application --workers 2 --timeout 120`
5. Add environment variables: `SECRET_KEY`, `DEBUG=False`, `TMDB_API_KEY`, `MODEL_DIR=models`

---

## Developer

**Srinivasan V**  
PSG College of Technology  
📧 [srinivasanv0401@gmail.com](mailto:srinivasanv0401@gmail.com)

---

## License

MIT License — free to use, modify, and distribute.
