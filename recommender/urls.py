from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = 'recommender'

urlpatterns = [
    # Main
    path('', views.main, name='main'),
    path('about/', views.about, name='about'),

    # Discovery — Phase 2
    path('movie/<str:title>/', views.movie_detail, name='movie_detail'),
    path('genres/', views.genre_browse, name='genre_browse'),
    path('trending/', views.trending, name='trending'),
    path('surprise/', views.surprise_me, name='surprise_me'),

    # Auth
    path('register/', views.register_view, name='register'),
    path('login/', auth_views.LoginView.as_view(template_name='recommender/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # User profile & personalization
    path('profile/', views.profile_view, name='profile'),
    path('profile/edit/', views.edit_profile_view, name='edit_profile'),
    path('user/<str:username>/', views.public_profile, name='public_profile'),

    # Watchlist
    path('watchlist/', views.watchlist_view, name='watchlist'),
    path('watchlist/add/', views.watchlist_add, name='watchlist_add'),
    path('watchlist/remove/', views.watchlist_remove, name='watchlist_remove'),

    # Ratings
    path('rate/', views.rate_movie, name='rate_movie'),
    path('ratings/', views.my_ratings, name='my_ratings'),

    # Watch history
    path('history/add/', views.history_add, name='history_add'),

    # Phase 3 — Reviews
    path('reviews/<str:title>/', views.movie_reviews, name='movie_reviews'),
    path('reviews/delete/<int:review_id>/', views.delete_review, name='delete_review'),
    path('reviews/like/<int:review_id>/', views.like_review, name='like_review'),
    path('reviews/comment/<int:review_id>/', views.add_comment, name='add_comment'),
    path('reviews/comment/delete/<int:comment_id>/', views.delete_comment, name='delete_comment'),

    # Phase 3 — Movie Lists
    path('lists/', views.lists_browse, name='lists_browse'),
    path('lists/mine/', views.my_lists, name='my_lists'),
    path('lists/create/', views.create_list, name='create_list'),
    path('lists/<slug:slug>/', views.list_detail, name='list_detail'),
    path('lists/<slug:slug>/add/', views.list_add_movie, name='list_add_movie'),
    path('lists/<slug:slug>/remove/', views.list_remove_movie, name='list_remove_movie'),
    path('lists/<slug:slug>/delete/', views.delete_list, name='delete_list'),

    # Phase 3 — Leaderboard
    path('leaderboard/', views.leaderboard, name='leaderboard'),

    # Phase 4 — Smart/AI Features
    path('mood/', views.mood_recommendations, name='mood_recommendations'),
    path('mood/<str:mood_key>/', views.mood_results, name='mood_results'),
    path('analytics/', views.analytics_dashboard, name='analytics_dashboard'),

    # Phase 5 — Share & Developer API
    path('share/<str:title>/', views.share_recommendations, name='share_recommendations'),

    # API endpoints
    path('api/search/', views.search_movies, name='search_movies'),
    path('api/model-status/', views.model_status, name='model_status'),
    path('api/health/', views.health_check, name='health_check'),
    path('api/recommendations/', views.api_recommendations, name='api_recommendations'),
]
