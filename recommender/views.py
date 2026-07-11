"""
Movie Recommendation System Views
Phase 1: Core recommendations + User accounts & personalization
Phase 2: Discovery — movie detail, genre browse, trending, surprise me
Phase 3: Social — reviews, comments, movie lists, leaderboard, public profiles
"""
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional
from difflib import get_close_matches

import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import json
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.db.models import Count, Avg
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from django.views.decorators.http import require_http_methods, require_POST
from urllib.parse import quote, quote_plus

from .models import Rating, WatchlistItem, WatchHistory, UserProfile, Review, ReviewComment, MovieList, MovieListItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TMDB live fallback — used when a movie isn't in the local dataset
# ---------------------------------------------------------------------------

TMDB_API_KEY = os.environ.get('TMDB_API_KEY', '')
TMDB_BASE    = 'https://api.themoviedb.org/3'


def _tmdb_key() -> str:
    """Read key at call time so env var set after import still works."""
    return os.environ.get('TMDB_API_KEY', TMDB_API_KEY)


def _tmdb_search_movie_id(title: str) -> Optional[int]:
    """Search TMDB for a movie title, return its TMDB id or None."""
    key = _tmdb_key()
    if not key:
        return None
    try:
        url = (f"{TMDB_BASE}/search/movie"
               f"?api_key={key}&query={urllib.request.quote(title)}&page=1")
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())
        results = data.get('results', [])
        return results[0]['id'] if results else None
    except Exception as e:
        logger.warning(f"TMDB search failed for '{title}': {e}")
        return None


def _tmdb_get_recommendations(tmdb_id: int, n: int = 15) -> List[Dict]:
    """Fetch similar movies from TMDB, returns all with original_language included."""
    key = _tmdb_key()
    if not key:
        return []
    try:
        url = (f"{TMDB_BASE}/movie/{tmdb_id}/similar"
               f"?api_key={key}&language=en-US&page=1")
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())
        movies = []
        for m in data.get('results', [])[:n * 2]:  # fetch more for language split
            poster = m.get('poster_path')
            title = m.get('title', '')
            movies.append({
                'title': title,
                'release_date': m.get('release_date', 'Unknown') or 'Unknown',
                'production': 'Unknown',
                'genres': 'N/A',
                'rating': f"{m['vote_average']:.1f}/10" if m.get('vote_average') else 'N/A',
                'votes': f"{m.get('vote_count', 0):,}",
                'similarity_score': f"{m.get('popularity', 0):.1f}",
                'imdb_id': None,
                'original_language': m.get('original_language', ''),
                'poster_url': f"https://image.tmdb.org/t/p/w500{poster}" if poster else None,
                'google_link': f"https://www.google.com/search?q={'+'.join(title.split())}+movie",
                'imdb_link': None,
                'justwatch_url': f"https://www.justwatch.com/us/search?q={quote_plus(title)}",
                'tmdb_id': m.get('id'),
            })
        return movies
    except Exception as e:
        logger.warning(f"TMDB similar failed for id {tmdb_id}: {e}")
        return []


def _tmdb_movie_detail(tmdb_id: int) -> Dict:
    """Fetch full movie detail from TMDB including watch providers."""
    key = _tmdb_key()
    if not key:
        return {}
    try:
        # Movie detail + watch providers in one append_to_response call
        url = (f"{TMDB_BASE}/movie/{tmdb_id}"
               f"?api_key={key}&language=en-US"
               f"&append_to_response=watch/providers")
        with urllib.request.urlopen(url, timeout=8) as r:
            d = json.loads(r.read())

        genres = ', '.join(g['name'] for g in d.get('genres', [])[:4]) or 'N/A'
        companies = d.get('production_companies', [])
        poster = d.get('poster_path')
        backdrop = d.get('backdrop_path')

        # Watch providers (US region)
        providers = []
        wp = d.get('watch/providers', {}).get('results', {}).get('US', {})
        for ptype in ('flatrate', 'rent', 'buy'):
            for p in wp.get(ptype, []):
                name = p.get('provider_name', '')
                logo = p.get('logo_path', '')
                if name and name not in [x['name'] for x in providers]:
                    providers.append({
                        'name': name,
                        'logo': f"https://image.tmdb.org/t/p/w45{logo}" if logo else None,
                        'type': ptype,
                    })

        return {
            'tmdb_id': tmdb_id,
            'title': d.get('title', ''),
            'tagline': d.get('tagline', ''),
            'overview': d.get('overview', ''),
            'release_date': d.get('release_date', ''),
            'runtime': d.get('runtime'),
            'genres': genres,
            'rating': f"{d['vote_average']:.1f}" if d.get('vote_average') else 'N/A',
            'votes': f"{d.get('vote_count', 0):,}",
            'original_language': d.get('original_language', ''),
            'production': companies[0]['name'] if companies else 'Unknown',
            'poster_url': f"https://image.tmdb.org/t/p/w500{poster}" if poster else None,
            'backdrop_url': f"https://image.tmdb.org/t/p/w1280{backdrop}" if backdrop else None,
            'imdb_id': d.get('imdb_id'),
            'imdb_link': f"https://www.imdb.com/title/{d['imdb_id']}" if d.get('imdb_id') else None,
            'justwatch_url': f"https://www.justwatch.com/us/search?q={quote_plus(d.get('title',''))}",
            'providers': providers,
        }
    except Exception as e:
        logger.warning(f"TMDB detail failed for id {tmdb_id}: {e}")
        return {}


def _tmdb_full_fallback(title: str, n: int = 15) -> Dict:
    """Full TMDB fallback: search for title then fetch similar movies split by language."""
    key = _tmdb_key()
    if not key:
        return {'error': 'TMDB key not configured'}

    tmdb_id = _tmdb_search_movie_id(title)
    if not tmdb_id:
        return {'error': f"Movie '{title}' not found in our database or TMDB."}

    detail = _tmdb_movie_detail(tmdb_id)
    source_lang = detail.get('original_language', '')
    source_info = {
        'production': detail.get('production', 'Unknown'),
        'rating': f"{detail.get('rating', 'N/A')}/10" if detail.get('rating') not in ('N/A', '') else 'N/A',
        'genres': detail.get('genres', 'N/A'),
    }

    all_recs = _tmdb_get_recommendations(tmdb_id, n)
    same_lang = [r for r in all_recs if r.get('original_language') == source_lang][:n]
    other_lang = [r for r in all_recs if r.get('original_language') != source_lang][:n]

    if not same_lang and not other_lang:
        return {'error': f"No similar movies found for '{title}'."}

    logger.info(f"TMDB fallback for '{title}' (id={tmdb_id}): {len(same_lang)} same-lang, {len(other_lang)} other-lang")
    return {
        'query_movie': title,
        'source_movie': source_info,
        'movie_detail': detail,
        'recommendations': same_lang if same_lang else other_lang,
        'other_lang_recommendations': other_lang if same_lang else [],
        'tmdb_fallback': True,
    }

_RECOMMENDER = None
_MODEL_LOADING = False
_MODEL_LOAD_PROGRESS = 0
_LOADING_THREAD = None
_LOAD_ERROR = None


class MovieRecommender:
    """Integrated recommender system matching training/infer.py logic"""

    def __init__(self, model_dir='models', progress_callback=None):
        self.model_dir = Path(model_dir)
        self.metadata = None
        # Top-K sparse format: indices (N,K) and scores (N,K)
        self.topk_indices = None
        self.topk_scores = None
        self.title_to_idx = None
        self.config = None
        self._load_models(progress_callback)

    def _load_models(self, progress_callback=None):
        global _MODEL_LOAD_PROGRESS
        logger.info(f"Loading models from {self.model_dir}...")

        if progress_callback:
            progress_callback(10)
        self.metadata = pd.read_parquet(self.model_dir / 'movie_metadata.parquet')
        if progress_callback:
            progress_callback(25)

        if progress_callback:
            progress_callback(40)
        topk_path = self.model_dir / 'top_k_similarities.npz'
        if topk_path.exists():
            data = np.load(topk_path)
            self.topk_indices = data['indices']   # shape (N, K)
            self.topk_scores  = data['scores']    # shape (N, K)
            logger.info(f"Loaded top-K similarity matrix: shape {self.topk_indices.shape}")
        else:
            # Fallback: load full matrix and convert on the fly (slow, high RAM)
            logger.warning("top_k_similarities.npz not found — falling back to full matrix")
            if (self.model_dir / 'similarity_matrix.npz').exists():
                full = load_npz(self.model_dir / 'similarity_matrix.npz').toarray()
            else:
                full = np.load(self.model_dir / 'similarity_matrix.npy')
            N, K = full.shape[0], 20
            self.topk_indices = np.empty((N, K), dtype=np.int32)
            self.topk_scores  = np.empty((N, K), dtype=np.float32)
            for i in range(N):
                row = full[i].astype(np.float32)
                row[i] = -np.inf
                top = np.argpartition(row, -K)[-K:]
                order = np.argsort(row[top])[::-1]
                self.topk_indices[i] = top[order]
                self.topk_scores[i]  = row[top[order]]
            del full
        if progress_callback:
            progress_callback(65)

        with open(self.model_dir / 'title_to_idx.json', 'r') as f:
            self.title_to_idx = json.load(f)
        if progress_callback:
            progress_callback(80)

        with open(self.model_dir / 'config.json', 'r') as f:
            self.config = json.load(f)
        if progress_callback:
            progress_callback(100)

        logger.info(f"Loaded {self.config['n_movies']:,} movies successfully")

    def find_movie(self, title: str) -> Optional[str]:
        # High cutoff so partial matches don't shadow TMDB fallback
        matches = get_close_matches(title, self.title_to_idx.keys(), n=1, cutoff=0.85)
        if matches:
            logger.info(f"find_movie: '{title}' matched local -> '{matches[0]}'")
        else:
            logger.info(f"find_movie: '{title}' not in local dataset, routing to TMDB")
        return matches[0] if matches else None

    def search_movies(self, query: str, n: int = 20) -> List[str]:
        query_lower = query.lower()
        return [title for title in self.title_to_idx.keys()
                if query_lower in title.lower()][:n]

    def get_recommendations(self, movie_title: str, n: int = 15, min_rating: float = None) -> Dict:
        matched_title = self.find_movie(movie_title)
        if not matched_title:
            # Try TMDB live fallback before giving up
            tmdb_result = _tmdb_full_fallback(movie_title, n)
            if 'error' not in tmdb_result:
                return tmdb_result
            return {'error': f"Movie '{movie_title}' not found", 'suggestions': self.search_movies(movie_title, 5)}

        movie_idx = self.title_to_idx[matched_title]
        source_movie = self.metadata.iloc[movie_idx]

        # Detect source language
        source_lang = None
        if 'original_language' in self.metadata.columns:
            source_lang = source_movie.get('original_language', None)
            if pd.isna(source_lang):
                source_lang = None

        # Use pre-computed top-K neighbors (already sorted, self excluded)
        neighbor_indices = self.topk_indices[movie_idx]
        neighbor_scores  = self.topk_scores[movie_idx]
        sim_scores = list(zip(neighbor_indices.tolist(), neighbor_scores.tolist()))

        same_lang = []
        other_lang = []

        for idx, score in sim_scores:
            movie = self.metadata.iloc[idx]
            if min_rating and movie['vote_average'] < min_rating:
                continue
            rec = {
                'title': movie['title'],
                'release_date': movie['release_date'] if pd.notna(movie['release_date']) else 'Unknown',
                'production': movie['primary_company'] if pd.notna(movie['primary_company']) else 'Unknown',
                'genres': ', '.join(list(movie['genres'])[:3]) if hasattr(movie['genres'], '__iter__') and not isinstance(movie['genres'], str) else 'N/A',
                'rating': f"{movie['vote_average']:.1f}/10" if pd.notna(movie['vote_average']) else 'N/A',
                'votes': f"{movie['vote_count']:,}" if pd.notna(movie['vote_count']) else 'N/A',
                'similarity_score': f"{score:.3f}",
                'imdb_id': movie['imdb_id'] if pd.notna(movie['imdb_id']) else None,
                'poster_url': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if pd.notna(movie['poster_path']) else None,
                'google_link': f"https://www.google.com/search?q={'+'.join(movie['title'].split())}+movie",
                'imdb_link': f"https://www.imdb.com/title/{movie['imdb_id']}" if pd.notna(movie['imdb_id']) else None,
            }
            # Split by language
            movie_lang = movie.get('original_language', None) if 'original_language' in self.metadata.columns else None
            if source_lang and not pd.isna(movie_lang) and movie_lang == source_lang:
                if len(same_lang) < n:
                    same_lang.append(rec)
            else:
                if len(other_lang) < n:
                    other_lang.append(rec)

            if len(same_lang) >= n and len(other_lang) >= n:
                break

        # If no same-lang split (language data missing), treat all as same
        recommendations = same_lang if same_lang else (other_lang or [])
        other_recommendations = other_lang if same_lang else []

        return {
            'query_movie': matched_title,
            'source_movie': {
                'production': source_movie['primary_company'] if pd.notna(source_movie['primary_company']) else 'Unknown',
                'rating': f"{source_movie['vote_average']:.1f}/10" if pd.notna(source_movie['vote_average']) else 'N/A',
                'genres': ', '.join(list(source_movie['genres'])[:3]) if hasattr(source_movie['genres'], '__iter__') and not isinstance(source_movie['genres'], str) else 'N/A',
            },
            'recommendations': recommendations,
            'other_lang_recommendations': other_recommendations,
            'source_lang': source_lang,
        }


def _load_model_in_background():
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    _MODEL_LOADING = True
    _MODEL_LOAD_PROGRESS = 0
    _LOAD_ERROR = None

    model_dir = getattr(settings, 'MODEL_DIR', os.environ.get('MODEL_DIR', 'models'))
    if not Path(model_dir).exists():
        model_dir = 'static'
        logger.warning("Model directory not found, using static directory")

    try:
        def progress_callback(progress):
            global _MODEL_LOAD_PROGRESS
            _MODEL_LOAD_PROGRESS = progress

        _RECOMMENDER = MovieRecommender(model_dir, progress_callback)
        _MODEL_LOADING = False
        _MODEL_LOAD_PROGRESS = 100
        logger.info("Model loaded successfully")
    except Exception as e:
        _MODEL_LOADING = False
        _LOAD_ERROR = str(e)
        logger.error(f"Failed to load recommender: {e}")


def _start_model_loading():
    global _LOADING_THREAD, _RECOMMENDER, _MODEL_LOADING
    if _RECOMMENDER is None and not _MODEL_LOADING:
        if _LOADING_THREAD is None or not _LOADING_THREAD.is_alive():
            _LOADING_THREAD = threading.Thread(target=_load_model_in_background, daemon=True)
            _LOADING_THREAD.start()


def _get_recommender():
    global _RECOMMENDER, _LOAD_ERROR
    if _RECOMMENDER is None:
        _start_model_loading()
        if _LOAD_ERROR:
            raise Exception(_LOAD_ERROR)
        return None
    return _RECOMMENDER


# ---------------------------------------------------------------------------
# Core recommendation views (unchanged from original)
# ---------------------------------------------------------------------------

@require_http_methods(["GET", "POST"])
def main(request):
    _start_model_loading()
    recommender = _get_recommender()

    if recommender is None:
        return render(request, 'recommender/index.html', {
            'all_movie_names': [],
            'total_movies': 0,
            'error_message': 'Model is still loading. Please wait a moment and try again.' if request.method == 'POST' else None,
        })

    titles_list = list(recommender.title_to_idx.keys())

    if request.method == 'GET':
        because_title, because_recs = _get_because_you_watched(request.user, recommender)

        # Trending fallback for guests or users with no history
        trending_movies = []
        if not because_recs:
            df = recommender.metadata
            min_votes = df['vote_count'].quantile(0.75)
            qualified = df[df['vote_count'] >= min_votes]
            top = qualified.sort_values('popularity', ascending=False).head(12)
            for _, m in top.iterrows():
                trending_movies.append({
                    'title': m['title'],
                    'rating': f"{m['vote_average']:.1f}" if pd.notna(m['vote_average']) else 'N/A',
                    'genres': ', '.join(list(m['genres'])[:2]) if hasattr(m['genres'], '__iter__') and not isinstance(m['genres'], str) else '',
                    'year': str(m['release_date'])[:4] if pd.notna(m['release_date']) else '',
                    'poster_url': f"https://image.tmdb.org/t/p/w185{m['poster_path']}" if pd.notna(m['poster_path']) else None,
                })

        return render(request, 'recommender/index.html', {
            'all_movie_names': titles_list,
            'total_movies': len(titles_list),
            'because_title': because_title,
            'because_recs': because_recs,
            'trending_movies': trending_movies,
        })

    movie_name = request.POST.get('movie_name', '').strip()
    if not movie_name:
        return render(request, 'recommender/index.html', {
            'all_movie_names': titles_list,
            'total_movies': len(titles_list),
            'error_message': 'Please enter a movie name.',
        })

    result = recommender.get_recommendations(movie_name, n=15)

    if 'error' in result:
        return render(request, 'recommender/index.html', {
            'all_movie_names': titles_list,
            'total_movies': len(titles_list),
            'input_movie_name': movie_name,
            'error_message': result['error'],
            'suggestions': result.get('suggestions', []),
        })

    # Track in watch history if logged in
    if request.user.is_authenticated:
        WatchHistory.objects.get_or_create(
            user=request.user,
            movie_title=result['query_movie'],
            defaults={
                'genres': result['source_movie'].get('genres', ''),
                'rating': result['source_movie'].get('rating', ''),
            }
        )

    # Annotate recs with user's watchlist/rating status
    watchlist_titles = set()
    rated_titles = {}
    if request.user.is_authenticated:
        watchlist_titles = set(
            WatchlistItem.objects.filter(user=request.user).values_list('movie_title', flat=True)
        )
        rated_titles = {
            r.movie_title: r.score
            for r in Rating.objects.filter(user=request.user)
        }

    for movie in result['recommendations']:
        movie['in_watchlist'] = movie['title'] in watchlist_titles
        movie['user_rating'] = rated_titles.get(movie['title'])
        movie['justwatch_url'] = f"https://www.justwatch.com/us/search?q={'+'.join(movie['title'].split())}"

    share_url = request.build_absolute_uri(f"/share/{quote(result['query_movie'])}/")
    return render(request, 'recommender/result.html', {
        'all_movie_names': titles_list,
        'input_movie_name': result['query_movie'],
        'source_movie': result['source_movie'],
        'recommended_movies': result['recommendations'],
        'other_lang_movies': result.get('other_lang_recommendations', []),
        'source_lang': result.get('source_lang', ''),
        'total_recommendations': len(result['recommendations']),
        'share_url': share_url,
    })


@require_http_methods(["GET"])
def search_movies(request):
    query = request.GET.get('q', '').strip()
    if len(query) < 2:
        return JsonResponse({'movies': [], 'count': 0})
    try:
        recommender = _get_recommender()
        if recommender is None:
            return JsonResponse({'movies': [], 'count': 0, 'loading': True})
        matching_movies = recommender.search_movies(query, n=20)
        return JsonResponse({'movies': matching_movies, 'count': len(matching_movies)})
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return JsonResponse({'error': 'Search failed'}, status=500)


@require_http_methods(["GET"])
def model_status(request):
    global _RECOMMENDER, _MODEL_LOADING, _MODEL_LOAD_PROGRESS, _LOAD_ERROR
    _start_model_loading()
    if _LOAD_ERROR:
        return JsonResponse({'loaded': False, 'progress': 0, 'status': 'error', 'error': _LOAD_ERROR})
    elif _RECOMMENDER is not None:
        return JsonResponse({'loaded': True, 'progress': 100, 'status': 'ready'})
    elif _MODEL_LOADING:
        return JsonResponse({'loaded': False, 'progress': _MODEL_LOAD_PROGRESS, 'status': 'loading'})
    else:
        return JsonResponse({'loaded': False, 'progress': 0, 'status': 'initializing'})


@require_http_methods(["GET"])
def movie_detail(request, title):
    """Movie detail page — poster, overview, full metadata, recommendations link"""
    recommender = _get_recommender()
    if recommender is None:
        return redirect('recommender:main')

    matched = recommender.find_movie(title)
    if not matched:
        messages.error(request, f'Movie "{title}" not found. Try searching for it.')
        return redirect('recommender:main')

    idx = recommender.title_to_idx[matched]
    m = recommender.metadata.iloc[idx]

    user_rating = None
    in_watchlist = False
    if request.user.is_authenticated:
        in_watchlist = WatchlistItem.objects.filter(user=request.user, movie_title=matched).exists()
        rating_obj = Rating.objects.filter(user=request.user, movie_title=matched).first()
        user_rating = rating_obj.score if rating_obj else None

    movie = {
        'title': m['title'],
        'overview': m['overview'] if pd.notna(m['overview']) else '',
        'release_date': m['release_date'] if pd.notna(m['release_date']) else 'Unknown',
        'production': m['primary_company'] if pd.notna(m['primary_company']) else 'Unknown',
        'genres': list(m['genres']) if hasattr(m['genres'], '__iter__') and not isinstance(m['genres'], str) else [],
        'rating': f"{m['vote_average']:.1f}" if pd.notna(m['vote_average']) else 'N/A',
        'votes': f"{m['vote_count']:,}" if pd.notna(m['vote_count']) else 'N/A',
        'popularity': f"{m['popularity']:.1f}" if pd.notna(m['popularity']) else 'N/A',
        'imdb_id': m['imdb_id'] if pd.notna(m['imdb_id']) else None,
        'poster_url': f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if pd.notna(m['poster_path']) else None,
        'google_link': f"https://www.google.com/search?q={'+'.join(m['title'].split())}+movie",
        'imdb_link': f"https://www.imdb.com/title/{m['imdb_id']}" if pd.notna(m['imdb_id']) else None,
        'in_watchlist': in_watchlist,
        'user_rating': user_rating,
    }
    return render(request, 'recommender/movie_detail.html', {'movie': movie})


def _tmdb_genre_movies(genre_name: str, page: int = 1, sort_by: str = 'rating',
                        language: str = '', decade: str = '') -> List[Dict]:
    """Fetch movies from TMDB by genre name, returns list of movie dicts."""
    key = _tmdb_key()
    if not key:
        return []

    # TMDB genre name -> id mapping
    TMDB_GENRE_IDS = {
        'action': 28, 'adventure': 12, 'animation': 16, 'comedy': 35,
        'crime': 80, 'documentary': 99, 'drama': 18, 'family': 10751,
        'fantasy': 14, 'history': 36, 'horror': 27, 'music': 10402,
        'mystery': 9648, 'romance': 10749, 'sciencefiction': 878,
        'science fiction': 878, 'thriller': 53, 'war': 10752, 'western': 37,
        'tv movie': 10770,
    }
    genre_id = TMDB_GENRE_IDS.get(genre_name.lower())
    if not genre_id:
        return []

    sort_map = {
        'rating': 'vote_average.desc',
        'popularity': 'popularity.desc',
        'year_desc': 'primary_release_date.desc',
        'year_asc': 'primary_release_date.asc',
    }
    tmdb_sort = sort_map.get(sort_by, 'vote_average.desc')

    params = (f"?api_key={key}&with_genres={genre_id}"
              f"&sort_by={tmdb_sort}&vote_count.gte=50&page={page}")
    if language:
        params += f"&with_original_language={language}"
    if decade:
        try:
            d = int(decade)
            params += f"&primary_release_date.gte={d}-01-01&primary_release_date.lte={d+9}-12-31"
        except ValueError:
            pass

    try:
        url = f"{TMDB_BASE}/discover/movie{params}"
        with urllib.request.urlopen(url, timeout=8) as r:
            data = json.loads(r.read())

        movies = []
        for m in data.get('results', []):
            poster = m.get('poster_path')
            title = m.get('title', '')
            release = m.get('release_date', '') or ''
            year = release[:4] if release else ''
            movies.append({
                'title': title,
                'rating': f"{m['vote_average']:.1f}" if m.get('vote_average') else 'N/A',
                'votes': f"{m.get('vote_count', 0):,}",
                'popularity': f"{m.get('popularity', 0):.0f}",
                'release_date': release,
                'year': year,
                'language': m.get('original_language', '').upper(),
                'poster_url': f"https://image.tmdb.org/t/p/w185{poster}" if poster else None,
                'genres': '',
                'overview_short': (m.get('overview', '')[:120] + '…') if len(m.get('overview', '')) > 120 else m.get('overview', ''),
                'imdb_id': None,
                'tmdb_id': m.get('id'),
            })
        return movies
    except Exception as e:
        logger.warning(f"TMDB genre browse failed for '{genre_name}': {e}")
        return []


@require_http_methods(["GET"])
def genre_browse(request):
    """
    Browse movies by genre with full filtering:
    - genre filter
    - sort by: rating | popularity | year (asc/desc)
    - decade filter
    - language filter
    - search within genre
    - pagination (24 per page)
    """
    recommender = _get_recommender()
    if recommender is None:
        return render(request, 'recommender/genre_browse.html', {
            'genres': [], 'movies': [],
            'selected_genre': None, 'page': 1, 'total_pages': 1,
            'total_count': 0, 'sort': 'rating', 'decade': '',
            'language': '', 'languages': [],
            'query': '', 'decades': [],
        })

    df = recommender.metadata

    # Reload metadata from disk if original_language column is missing
    # (added after initial load via the CSV enrichment step)
    if 'original_language' not in df.columns:
        import importlib
        from pathlib import Path
        model_dir = Path(getattr(settings, 'MODEL_DIR', 'models'))
        df = pd.read_parquet(model_dir / 'movie_metadata.parquet')
        recommender.metadata = df

    # --- Build genre list with counts ---
    genre_counts = {}
    for g_list in df['genres']:
        if hasattr(g_list, '__iter__') and not isinstance(g_list, str):
            for g in g_list:
                genre_counts[g] = genre_counts.get(g, 0) + 1
    all_genres = sorted(genre_counts.items(), key=lambda x: x[0])

    # --- Build language lookup (code -> display name) ---
    LANG_NAMES = {
        'en': 'English', 'fr': 'French', 'it': 'Italian', 'ja': 'Japanese',
        'es': 'Spanish', 'de': 'German', 'hi': 'Hindi', 'ko': 'Korean',
        'zh': 'Chinese', 'ru': 'Russian', 'pt': 'Portuguese', 'ar': 'Arabic',
        'tr': 'Turkish', 'sv': 'Swedish', 'nl': 'Dutch', 'pl': 'Polish',
        'da': 'Danish', 'fi': 'Finnish', 'no': 'Norwegian', 'th': 'Thai',
        'id': 'Indonesian', 'ro': 'Romanian', 'cs': 'Czech', 'hu': 'Hungarian',
        'he': 'Hebrew', 'el': 'Greek', 'uk': 'Ukrainian', 'fa': 'Persian',
        'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam',
    }

    # --- Read filter params ---
    selected_genre = request.GET.get('genre', '').strip().lower()
    sort_by = request.GET.get('sort', 'rating')
    decade_filter = request.GET.get('decade', '').strip()
    language_filter = request.GET.get('language', '').strip().lower()
    query = request.GET.get('q', '').strip().lower()
    try:
        page = max(1, int(request.GET.get('page', 1)))
    except ValueError:
        page = 1

    PAGE_SIZE = 24
    movies = []
    total_pages = 1
    total_count = 0
    decades = []
    languages = []

    if selected_genre:
        # ── TMDB live results (primary source — always up to date) ──
        tmdb_movies = _tmdb_genre_movies(
            selected_genre, page=page, sort_by=sort_by,
            language=language_filter, decade=decade_filter
        )

        if tmdb_movies:
            movies = tmdb_movies
            # TMDB returns 20 per page with up to 500 pages
            total_count = 500 * 20  # approximate
            total_pages = 500
            decades = list(range(1920, 2030, 10))
            languages = [
                ('en', 'English', ''), ('ta', 'Tamil', ''), ('hi', 'Hindi', ''),
                ('ko', 'Korean', ''), ('ja', 'Japanese', ''), ('fr', 'French', ''),
                ('es', 'Spanish', ''), ('de', 'German', ''), ('it', 'Italian', ''),
                ('te', 'Telugu', ''), ('ml', 'Malayalam', ''), ('zh', 'Chinese', ''),
            ]
        else:
            # Fallback to local dataset if TMDB unavailable
            mask = df['genres'].apply(
                lambda g: hasattr(g, '__iter__') and not isinstance(g, str) and selected_genre in g
            )
            filtered = df[mask].copy()

            if 'original_language' in filtered.columns:
                lang_counts = filtered['original_language'].value_counts()
                languages = [
                    (code, LANG_NAMES.get(code, code.upper()), count)
                    for code, count in lang_counts.items()
                    if pd.notna(code)
                ]

            if language_filter and 'original_language' in filtered.columns:
                filtered = filtered[filtered['original_language'] == language_filter]

            def _year(rd):
                try:
                    return int(str(rd)[:4]) if pd.notna(rd) and str(rd)[:4].isdigit() else 0
                except Exception:
                    return 0

            filtered['_year'] = filtered['release_date'].apply(_year)
            valid_years = filtered['_year'][filtered['_year'] > 0]
            if not valid_years.empty:
                min_dec = (valid_years.min() // 10) * 10
                max_dec = (valid_years.max() // 10) * 10
                decades = list(range(min_dec, max_dec + 10, 10))

            if decade_filter:
                try:
                    d = int(decade_filter)
                    filtered = filtered[(filtered['_year'] >= d) & (filtered['_year'] < d + 10)]
                except ValueError:
                    pass

            if query:
                filtered = filtered[filtered['title'].str.lower().str.contains(query, na=False)]

            if sort_by == 'popularity':
                filtered = filtered.sort_values('popularity', ascending=False)
            elif sort_by == 'year_desc':
                filtered = filtered.sort_values('_year', ascending=False)
            elif sort_by == 'year_asc':
                filtered = filtered[filtered['_year'] > 0].sort_values('_year', ascending=True)
            else:
                filtered = filtered.sort_values('vote_average', ascending=False)

            total_count = len(filtered)
            total_pages = max(1, (total_count + PAGE_SIZE - 1) // PAGE_SIZE)
            page = min(page, total_pages)
            start = (page - 1) * PAGE_SIZE
            page_slice = filtered.iloc[start:start + PAGE_SIZE]

            for _, m in page_slice.iterrows():
                year = str(m['_year']) if m['_year'] > 0 else ''
                lang_code = m.get('original_language', '') if 'original_language' in m.index else ''
                movies.append({
                    'title': m['title'],
                    'rating': f"{m['vote_average']:.1f}" if pd.notna(m['vote_average']) else 'N/A',
                    'votes': f"{int(m['vote_count']):,}" if pd.notna(m['vote_count']) else '',
                    'popularity': f"{m['popularity']:.0f}" if pd.notna(m['popularity']) else '',
                    'release_date': m['release_date'] if pd.notna(m['release_date']) else '',
                    'year': year,
                    'language': LANG_NAMES.get(lang_code, lang_code.upper()) if lang_code else '',
                    'poster_url': f"https://image.tmdb.org/t/p/w185{m['poster_path']}" if pd.notna(m['poster_path']) else None,
                    'genres': ', '.join(list(m['genres'])[:3]) if hasattr(m['genres'], '__iter__') and not isinstance(m['genres'], str) else '',
                    'overview_short': (m['overview'][:120] + '…') if pd.notna(m['overview']) and len(str(m['overview'])) > 120 else (m['overview'] if pd.notna(m['overview']) else ''),
                    'imdb_id': m['imdb_id'] if pd.notna(m['imdb_id']) else None,
                })

    page_range = _page_range(page, total_pages)

    return render(request, 'recommender/genre_browse.html', {
        'genres': all_genres,
        'movies': movies,
        'selected_genre': selected_genre,
        'sort': sort_by,
        'decade': decade_filter,
        'decades': decades,
        'language': language_filter,
        'languages': languages,
        'query': query,
        'page': page,
        'total_pages': total_pages,
        'total_count': total_count,
        'page_range': page_range,
    })


def _page_range(current, total, window=2):
    """Return a compact list of page numbers with None as ellipsis sentinel."""
    if total <= 1:
        return []
    pages = set()
    pages.add(1)
    pages.add(total)
    for i in range(max(1, current - window), min(total, current + window) + 1):
        pages.add(i)
    result = []
    prev = None
    for p in sorted(pages):
        if prev is not None and p - prev > 1:
            result.append(None)  # ellipsis
        result.append(p)
        prev = p
    return result


@require_http_methods(["GET"])
def trending(request):
    """Top-rated & most popular movies — popularity-based fallback for new users"""
    recommender = _get_recommender()
    if recommender is None:
        return render(request, 'recommender/trending.html', {'top_rated': [], 'most_popular': []})

    df = recommender.metadata

    def _serialize(row):
        m = row[1]
        return {
            'title': m['title'],
            'rating': f"{m['vote_average']:.1f}" if pd.notna(m['vote_average']) else 'N/A',
            'votes': f"{m['vote_count']:,}" if pd.notna(m['vote_count']) else 'N/A',
            'release_date': m['release_date'] if pd.notna(m['release_date']) else '',
            'genres': ', '.join(list(m['genres'])[:3]) if hasattr(m['genres'], '__iter__') and not isinstance(m['genres'], str) else '',
            'poster_url': f"https://image.tmdb.org/t/p/w185{m['poster_path']}" if pd.notna(m['poster_path']) else None,
            'imdb_id': m['imdb_id'] if pd.notna(m['imdb_id']) else None,
            'google_link': f"https://www.google.com/search?q={'+'.join(m['title'].split())}+movie",
        }

    # Minimum vote threshold for credibility
    min_votes = df['vote_count'].quantile(0.70)
    qualified = df[df['vote_count'] >= min_votes]

    top_rated = [_serialize(r) for r in qualified.sort_values('vote_average', ascending=False).head(96).iterrows()]
    most_popular = [_serialize(r) for r in qualified.sort_values('popularity', ascending=False).head(96).iterrows()]

    return render(request, 'recommender/trending.html', {
        'top_rated': top_rated[:24],
        'most_popular': most_popular[:24],
        'top_rated_json': json.dumps(top_rated),
        'most_popular_json': json.dumps(most_popular),
    })


@require_http_methods(["GET"])
def surprise_me(request):
    """Pick a random well-rated movie and redirect to its recommendations"""
    recommender = _get_recommender()
    if recommender is None:
        return redirect('recommender:main')

    df = recommender.metadata
    min_votes = df['vote_count'].quantile(0.60)
    pool = df[(df['vote_count'] >= min_votes) & (df['vote_average'] >= 6.5)]
    if pool.empty:
        pool = df
    pick = pool.sample(1).iloc[0]
    title = pick['title']
    return redirect(reverse('recommender:movie_detail', kwargs={'title': title}))


@require_http_methods(["GET"])
def health_check(request):
    try:
        recommender = _get_recommender()
        return JsonResponse({
            'status': 'healthy',
            'movies_loaded': recommender.config['n_movies'],
            'model_dir': str(recommender.model_dir),
            'model_loaded': True,
        })
    except Exception as e:
        return JsonResponse({'status': 'unhealthy', 'error': str(e)}, status=503)


# ---------------------------------------------------------------------------
# Phase 1 — Auth & Account views
# ---------------------------------------------------------------------------

def register_view(request):
    if request.user.is_authenticated:
        return redirect('recommender:main')

    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, f'Welcome, {user.username}! Your account is ready.')
            return redirect('recommender:main')
    else:
        form = UserCreationForm()

    return render(request, 'recommender/register.html', {'form': form})


@login_required
def profile_view(request):
    watchlist = WatchlistItem.objects.filter(user=request.user)
    ratings = Rating.objects.filter(user=request.user)
    history = WatchHistory.objects.filter(user=request.user)[:20]
    return render(request, 'recommender/profile.html', {
        'watchlist': watchlist,
        'ratings': ratings,
        'history': history,
    })


@login_required
def edit_profile_view(request):
    profile = request.user.profile
    if request.method == 'POST':
        profile.bio = request.POST.get('bio', '').strip()[:500]
        genres_raw = request.POST.get('favorite_genres', '')
        profile.favorite_genres = [g.strip() for g in genres_raw.split(',') if g.strip()]
        profile.save()
        messages.success(request, 'Profile updated.')
        return redirect('recommender:profile')
    return render(request, 'recommender/edit_profile.html', {'profile': profile})


# ---------------------------------------------------------------------------
# Watchlist views
# ---------------------------------------------------------------------------

@login_required
def watchlist_view(request):
    items = WatchlistItem.objects.filter(user=request.user)
    return render(request, 'recommender/watchlist.html', {'watchlist': items})


@login_required
@require_POST
def watchlist_add(request):
    title = request.POST.get('movie_title', '').strip()
    if not title:
        return JsonResponse({'error': 'No title provided'}, status=400)

    item, created = WatchlistItem.objects.get_or_create(
        user=request.user,
        movie_title=title,
        defaults={
            'imdb_id': request.POST.get('imdb_id', ''),
            'poster_url': request.POST.get('poster_url', ''),
            'genres': request.POST.get('genres', ''),
            'rating': request.POST.get('rating', ''),
        }
    )
    return JsonResponse({'added': created, 'title': title})


@login_required
@require_POST
def watchlist_remove(request):
    title = request.POST.get('movie_title', '').strip()
    deleted, _ = WatchlistItem.objects.filter(user=request.user, movie_title=title).delete()
    return JsonResponse({'removed': deleted > 0, 'title': title})


# ---------------------------------------------------------------------------
# Rating views
# ---------------------------------------------------------------------------

@login_required
@require_POST
def rate_movie(request):
    title = request.POST.get('movie_title', '').strip()
    score_raw = request.POST.get('score', '')
    review = request.POST.get('review', '').strip()[:1000]

    if not title or not score_raw:
        return JsonResponse({'error': 'Title and score required'}, status=400)

    try:
        score = int(score_raw)
        if not 1 <= score <= 10:
            raise ValueError
    except ValueError:
        return JsonResponse({'error': 'Score must be 1–10'}, status=400)

    rating, created = Rating.objects.update_or_create(
        user=request.user,
        movie_title=title,
        defaults={'score': score, 'review': review},
    )
    return JsonResponse({'saved': True, 'created': created, 'score': score, 'title': title})


@login_required
def my_ratings(request):
    ratings = Rating.objects.filter(user=request.user)
    return render(request, 'recommender/my_ratings.html', {'ratings': ratings})


# ---------------------------------------------------------------------------
# Watch history
# ---------------------------------------------------------------------------

@login_required
@require_POST
def history_add(request):
    title = request.POST.get('movie_title', '').strip()
    if not title:
        return JsonResponse({'error': 'No title provided'}, status=400)
    item, created = WatchHistory.objects.get_or_create(
        user=request.user,
        movie_title=title,
        defaults={
            'imdb_id': request.POST.get('imdb_id', ''),
            'poster_url': request.POST.get('poster_url', ''),
            'genres': request.POST.get('genres', ''),
            'rating': request.POST.get('rating', ''),
        }
    )
    return JsonResponse({'added': created, 'title': title})


# ---------------------------------------------------------------------------
# Phase 3 — Social & Engagement
# ---------------------------------------------------------------------------

# ── Reviews ──

@require_http_methods(["GET", "POST"])
def movie_reviews(request, title):
    """All reviews for a movie + submit form."""
    reviews = Review.objects.filter(movie_title=title).select_related('user').prefetch_related('comments__user', 'likes')
    user_review = None
    if request.user.is_authenticated:
        user_review = Review.objects.filter(user=request.user, movie_title=title).first()

    if request.method == 'POST':
        if not request.user.is_authenticated:
            return redirect(f'/login/?next=/reviews/{title}/')
        body = request.POST.get('body', '').strip()
        score_raw = request.POST.get('score', '').strip()
        if not body:
            messages.error(request, 'Review body cannot be empty.')
            return redirect('recommender:movie_reviews', title=title)
        score = None
        if score_raw:
            try:
                score = int(score_raw)
                if not 1 <= score <= 10:
                    score = None
            except ValueError:
                score = None
        Review.objects.update_or_create(
            user=request.user, movie_title=title,
            defaults={'body': body, 'score': score}
        )
        messages.success(request, 'Review saved.')
        return redirect('recommender:movie_reviews', title=title)

    return render(request, 'recommender/reviews.html', {
        'movie_title': title,
        'reviews': reviews,
        'user_review': user_review,
    })


@login_required
@require_POST
def delete_review(request, review_id):
    review = get_object_or_404(Review, id=review_id, user=request.user)
    title = review.movie_title
    review.delete()
    messages.success(request, 'Review deleted.')
    return redirect('recommender:movie_reviews', title=title)


@login_required
@require_POST
def like_review(request, review_id):
    review = get_object_or_404(Review, id=review_id)
    if request.user in review.likes.all():
        review.likes.remove(request.user)
        liked = False
    else:
        review.likes.add(request.user)
        liked = True
    return JsonResponse({'liked': liked, 'count': review.like_count})


@login_required
@require_POST
def add_comment(request, review_id):
    review = get_object_or_404(Review, id=review_id)
    body = request.POST.get('body', '').strip()[:500]
    if not body:
        return JsonResponse({'error': 'Empty comment'}, status=400)
    comment = ReviewComment.objects.create(review=review, user=request.user, body=body)
    return JsonResponse({
        'id': comment.id,
        'user': comment.user.username,
        'body': comment.body,
        'created_at': comment.created_at.strftime('%b %d, %Y'),
    })


@login_required
@require_POST
def delete_comment(request, comment_id):
    comment = get_object_or_404(ReviewComment, id=comment_id, user=request.user)
    comment.delete()
    return JsonResponse({'deleted': True})


# ── Movie Lists ──

@require_http_methods(["GET"])
def lists_browse(request):
    """All public community lists."""
    lists = MovieList.objects.filter(is_public=True).select_related('owner').annotate(
        item_count=Count('items')
    ).order_by('-updated_at')
    return render(request, 'recommender/lists_browse.html', {'lists': lists})


@login_required
def my_lists(request):
    lists = MovieList.objects.filter(owner=request.user).annotate(item_count=Count('items'))
    return render(request, 'recommender/my_lists.html', {'lists': lists})


@login_required
@require_http_methods(["GET", "POST"])
def create_list(request):
    if request.method == 'POST':
        title = request.POST.get('title', '').strip()[:150]
        description = request.POST.get('description', '').strip()[:500]
        is_public = request.POST.get('is_public') == 'on'
        if not title:
            messages.error(request, 'List title is required.')
            return render(request, 'recommender/create_list.html')
        ml = MovieList.objects.create(owner=request.user, title=title, description=description, is_public=is_public)
        messages.success(request, f'List "{ml.title}" created.')
        return redirect('recommender:list_detail', slug=ml.slug)
    return render(request, 'recommender/create_list.html')


@require_http_methods(["GET"])
def list_detail(request, slug):
    ml = get_object_or_404(MovieList, slug=slug)
    if not ml.is_public and ml.owner != request.user:
        return redirect('recommender:lists_browse')
    items = ml.items.all()
    return render(request, 'recommender/list_detail.html', {'movie_list': ml, 'items': items})


@login_required
@require_POST
def list_add_movie(request, slug):
    ml = get_object_or_404(MovieList, slug=slug, owner=request.user)
    title = request.POST.get('movie_title', '').strip()
    note = request.POST.get('note', '').strip()[:255]
    poster = request.POST.get('poster_url', '').strip()
    if not title:
        return JsonResponse({'error': 'No title'}, status=400)
    item, created = MovieListItem.objects.get_or_create(
        movie_list=ml, movie_title=title,
        defaults={'note': note, 'poster_url': poster}
    )
    return JsonResponse({'added': created, 'title': title})


@login_required
@require_POST
def list_remove_movie(request, slug):
    ml = get_object_or_404(MovieList, slug=slug, owner=request.user)
    title = request.POST.get('movie_title', '').strip()
    deleted, _ = MovieListItem.objects.filter(movie_list=ml, movie_title=title).delete()
    return JsonResponse({'removed': deleted > 0})


@login_required
@require_POST
def delete_list(request, slug):
    ml = get_object_or_404(MovieList, slug=slug, owner=request.user)
    ml.delete()
    messages.success(request, 'List deleted.')
    return redirect('recommender:my_lists')


# ── Leaderboard ──

@require_http_methods(["GET"])
def leaderboard(request):
    """Top reviewers by review count and avg score."""
    top_reviewers = (
        User.objects.annotate(
            review_count=Count('reviews'),
            avg_score=Avg('reviews__score'),
            rating_count=Count('ratings'),
        )
        .filter(review_count__gt=0)
        .order_by('-review_count')[:20]
    )
    # Most reviewed movies
    top_movies = (
        Review.objects.values('movie_title')
        .annotate(count=Count('id'), avg=Avg('score'))
        .order_by('-count')[:10]
    )
    return render(request, 'recommender/leaderboard.html', {
        'top_reviewers': top_reviewers,
        'top_movies': top_movies,
    })


# ── Public user profile ──

@require_http_methods(["GET"])
def public_profile(request, username):
    profile_user = get_object_or_404(User, username=username)
    reviews = Review.objects.filter(user=profile_user).order_by('-created_at')[:10]
    public_lists = MovieList.objects.filter(owner=profile_user, is_public=True).annotate(item_count=Count('items'))
    ratings = Rating.objects.filter(user=profile_user)
    return render(request, 'recommender/public_profile.html', {
        'profile_user': profile_user,
        'reviews': reviews,
        'public_lists': public_lists,
        'ratings': ratings,
    })


# ---------------------------------------------------------------------------
# Phase 4 — Smart / AI-Powered Enhancements
# ---------------------------------------------------------------------------

# Mood → genre weights + overview keywords
MOOD_CONFIG = {
    'feel-good': {
        'label': 'Feel-Good',
        'emoji': '😊',
        'desc': 'Uplifting, warm, leaves you smiling',
        'genres': ['comedy', 'family', 'animation', 'romance'],
        'keywords': ['uplifting', 'heartwarming', 'inspiring', 'joy', 'fun', 'happy', 'laugh', 'warm'],
        'min_rating': 6.5,
    },
    'thrilling': {
        'label': 'Thrilling',
        'emoji': '🤯',
        'desc': 'Edge-of-your-seat suspense and action',
        'genres': ['thriller', 'action', 'crime'],
        'keywords': ['suspense', 'tense', 'chase', 'danger', 'escape', 'assassin', 'spy', 'twist'],
        'min_rating': 6.0,
    },
    'thought-provoking': {
        'label': 'Thought-Provoking',
        'emoji': '🤔',
        'desc': 'Deep, cerebral, makes you question things',
        'genres': ['drama', 'sciencefiction', 'documentary'],
        'keywords': ['philosophy', 'identity', 'reality', 'future', 'society', 'moral', 'ethics', 'mind'],
        'min_rating': 7.0,
    },
    'scary': {
        'label': 'Scary',
        'emoji': '👻',
        'desc': 'Horror, tension and things that go bump',
        'genres': ['horror', 'thriller'],
        'keywords': ['fear', 'haunted', 'ghost', 'monster', 'nightmare', 'terror', 'evil', 'paranormal'],
        'min_rating': 5.5,
    },
    'adventurous': {
        'label': 'Adventurous',
        'emoji': '🌍',
        'desc': 'Epic journeys, exploration and discovery',
        'genres': ['adventure', 'action', 'fantasy'],
        'keywords': ['journey', 'quest', 'explore', 'discover', 'treasure', 'hero', 'epic', 'battle'],
        'min_rating': 6.0,
    },
    'romantic': {
        'label': 'Romantic',
        'emoji': '❤️',
        'desc': 'Love stories and emotional connections',
        'genres': ['romance', 'drama'],
        'keywords': ['love', 'relationship', 'romance', 'affair', 'heartbreak', 'passion', 'couple'],
        'min_rating': 6.0,
    },
    'funny': {
        'label': 'Funny',
        'emoji': '😂',
        'desc': 'Pure laughs, comedy and absurdity',
        'genres': ['comedy'],
        'keywords': ['comedy', 'hilarious', 'satire', 'parody', 'humor', 'laughs', 'witty', 'absurd'],
        'min_rating': 5.5,
    },
    'dark': {
        'label': 'Dark & Intense',
        'emoji': '🖤',
        'desc': 'Gritty, dark themes, hard-hitting drama',
        'genres': ['crime', 'drama', 'thriller'],
        'keywords': ['dark', 'gritty', 'brutal', 'violence', 'corrupt', 'crime', 'tragedy', 'war'],
        'min_rating': 6.5,
    },
}


@require_http_methods(["GET"])
def mood_recommendations(request):
    """Landing page showing all mood cards."""
    moods = [{'key': k, **v} for k, v in MOOD_CONFIG.items()]
    return render(request, 'recommender/mood.html', {'moods': moods})


@require_http_methods(["GET"])
def mood_results(request, mood_key):
    """Movies filtered by a specific mood."""
    if mood_key not in MOOD_CONFIG:
        return redirect('recommender:mood_recommendations')

    mood = MOOD_CONFIG[mood_key]
    recommender = _get_recommender()
    if recommender is None:
        return redirect('recommender:main')

    df = recommender.metadata
    target_genres = set(mood['genres'])
    keywords = mood['keywords']
    min_rating = mood.get('min_rating', 6.0)

    # Score each movie: genre overlap + keyword hits in overview
    def _score(row):
        g = row['genres']
        genres_set = set(g) if hasattr(g, '__iter__') and not isinstance(g, str) else set()
        genre_score = len(genres_set & target_genres) * 3

        overview = str(row['overview']).lower() if pd.notna(row['overview']) else ''
        keyword_score = sum(1 for kw in keywords if kw in overview)

        return genre_score + keyword_score

    df = df[df['vote_average'] >= min_rating].copy()
    min_votes = df['vote_count'].quantile(0.4)
    df = df[df['vote_count'] >= min_votes].copy()

    df['_mood_score'] = df.apply(_score, axis=1)
    results = df[df['_mood_score'] > 0].sort_values(
        ['_mood_score', 'vote_average'], ascending=[False, False]
    ).head(24)

    movies = []
    for _, m in results.iterrows():
        movies.append({
            'title': m['title'],
            'rating': f"{m['vote_average']:.1f}" if pd.notna(m['vote_average']) else 'N/A',
            'votes': f"{int(m['vote_count']):,}" if pd.notna(m['vote_count']) else '',
            'year': str(m['release_date'])[:4] if pd.notna(m['release_date']) else '',
            'genres': ', '.join(list(m['genres'])[:3]) if hasattr(m['genres'], '__iter__') and not isinstance(m['genres'], str) else '',
            'overview_short': (str(m['overview'])[:130] + '…') if pd.notna(m['overview']) and len(str(m['overview'])) > 130 else (m['overview'] if pd.notna(m['overview']) else ''),
            'poster_url': f"https://image.tmdb.org/t/p/w185{m['poster_path']}" if pd.notna(m['poster_path']) else None,
        })

    return render(request, 'recommender/mood_results.html', {
        'mood': mood,
        'mood_key': mood_key,
        'movies': movies,
        'all_moods': [{'key': k, 'label': v['label'], 'emoji': v['emoji']} for k, v in MOOD_CONFIG.items()],
    })


def _get_because_you_watched(user, recommender, limit=8):
    """Return recommendations based on user's most recent watch history entry."""
    if not user.is_authenticated or recommender is None:
        return None, []

    history = WatchHistory.objects.filter(user=user).order_by('-watched_at').first()
    if not history:
        return None, []

    result = recommender.get_recommendations(history.movie_title, n=limit)
    if 'error' in result or not result.get('recommendations'):
        return None, []

    return history.movie_title, result['recommendations']


# ── Admin Analytics Dashboard ──

@login_required
def analytics_dashboard(request):
    """Admin-only analytics dashboard."""
    if not request.user.is_staff:
        return redirect('recommender:main')

    from django.contrib.auth.models import User as AuthUser
    from django.utils import timezone
    from datetime import timedelta

    now = timezone.now()
    week_ago = now - timedelta(days=7)
    month_ago = now - timedelta(days=30)

    # User stats
    total_users = AuthUser.objects.count()
    new_users_week = AuthUser.objects.filter(date_joined__gte=week_ago).count()
    new_users_month = AuthUser.objects.filter(date_joined__gte=month_ago).count()

    # Content stats
    total_reviews = Review.objects.count()
    total_ratings = Rating.objects.count()
    total_watchlist = WatchlistItem.objects.count()
    total_lists = MovieList.objects.count()

    # Most searched (watch history = proxy for searches)
    most_searched = (
        WatchHistory.objects.values('movie_title')
        .annotate(count=Count('id'))
        .order_by('-count')[:10]
    )

    # Most rated movies
    most_rated = (
        Rating.objects.values('movie_title')
        .annotate(count=Count('id'), avg=Avg('score'))
        .order_by('-count')[:10]
    )

    # Most reviewed movies
    most_reviewed = (
        Review.objects.values('movie_title')
        .annotate(count=Count('id'))
        .order_by('-count')[:10]
    )

    # Genre popularity from watchlist
    genre_popularity = (
        WatchlistItem.objects.exclude(genres='')
        .values('genres')
        .annotate(count=Count('id'))
        .order_by('-count')[:10]
    )

    # Recent signups
    recent_users = AuthUser.objects.order_by('-date_joined')[:10]

    return render(request, 'recommender/analytics.html', {
        'total_users': total_users,
        'new_users_week': new_users_week,
        'new_users_month': new_users_month,
        'total_reviews': total_reviews,
        'total_ratings': total_ratings,
        'total_watchlist': total_watchlist,
        'total_lists': total_lists,
        'most_searched': most_searched,
        'most_rated': most_rated,
        'most_reviewed': most_reviewed,
        'genre_popularity': genre_popularity,
        'recent_users': recent_users,
    })


# ---------------------------------------------------------------------------
# Phase 5 — UX / Scale / Monetization
# ---------------------------------------------------------------------------

def _justwatch_url(title):
    """Generate JustWatch search URL for a movie title."""
    return f"https://www.justwatch.com/us/search?q={quote_plus(title)}"

@require_http_methods(["GET"])
def share_recommendations(request, title):
    """Shareable recommendations page — same as result but via GET with a clean URL."""
    recommender = _get_recommender()
    if recommender is None:
        return redirect('recommender:main')

    result = recommender.get_recommendations(title, n=15)
    if 'error' in result:
        return redirect('recommender:main')

    watchlist_titles = set()
    rated_titles = {}
    if request.user.is_authenticated:
        watchlist_titles = set(
            WatchlistItem.objects.filter(user=request.user).values_list('movie_title', flat=True)
        )
        rated_titles = {r.movie_title: r.score for r in Rating.objects.filter(user=request.user)}

    for movie in result['recommendations']:
        movie['in_watchlist'] = movie['title'] in watchlist_titles
        movie['user_rating'] = rated_titles.get(movie['title'])
        movie['justwatch_url'] = _justwatch_url(movie['title'])

    share_url = request.build_absolute_uri()
    return render(request, 'recommender/result.html', {
        'all_movie_names': list(recommender.title_to_idx.keys()),
        'input_movie_name': result['query_movie'],
        'source_movie': result['source_movie'],
        'movie_detail': result.get('movie_detail'),
        'recommended_movies': result['recommendations'],
        'other_lang_movies': result.get('other_lang_recommendations', []),
        'total_recommendations': len(result['recommendations']),
        'share_url': share_url,
        'is_shared': True,
    })


@require_http_methods(["GET"])
def tmdb_movie_detail(request, tmdb_id):
    """Full movie detail page using TMDB API — poster, overview, watch providers, similar movies."""
    key = _tmdb_key()
    if not key:
        return redirect('recommender:main')

    detail = _tmdb_movie_detail(int(tmdb_id))
    if not detail:
        return redirect('recommender:main')

    source_lang = detail.get('original_language', '')
    all_recs = _tmdb_get_recommendations(int(tmdb_id), n=15)
    same_lang = [r for r in all_recs if r.get('original_language') == source_lang][:15]
    other_lang = [r for r in all_recs if r.get('original_language') != source_lang][:15]

    return render(request, 'recommender/tmdb_movie_detail.html', {
        'movie': detail,
        'same_lang_movies': same_lang,
        'other_lang_movies': other_lang,
    })


# ── Developer JSON API ──

@require_http_methods(["GET"])
def api_recommendations(request):
    """
    Public JSON API: GET /api/recommendations/?movie=<title>&n=<count>
    Returns up to 20 recommendations. No auth required.
    """
    title = request.GET.get('movie', '').strip()
    try:
        n = min(int(request.GET.get('n', 10)), 20)
    except ValueError:
        n = 10

    if not title:
        return JsonResponse({'error': 'movie parameter required'}, status=400)

    recommender = _get_recommender()
    if recommender is None:
        return JsonResponse({'error': 'Model still loading, try again shortly'}, status=503)

    result = recommender.get_recommendations(title, n=n)
    if 'error' in result:
        return JsonResponse({'error': result['error'], 'suggestions': result.get('suggestions', [])}, status=404)

    return JsonResponse({
        'query': result['query_movie'],
        'count': len(result['recommendations']),
        'recommendations': [
            {
                'title': m['title'],
                'rating': m['rating'],
                'genres': m['genres'],
                'release_date': m['release_date'],
                'similarity_score': m['similarity_score'],
                'imdb_id': m.get('imdb_id'),
                'poster_url': m.get('poster_url'),
            }
            for m in result['recommendations']
        ],
    })


@require_http_methods(["GET"])
def about(request):
    return render(request, 'recommender/about.html')
