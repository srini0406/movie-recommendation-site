from django.contrib import admin
from .models import UserProfile, Rating, WatchlistItem, WatchHistory, Review, ReviewComment, MovieList, MovieListItem


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'created_at')
    search_fields = ('user__username', 'user__email')


@admin.register(Rating)
class RatingAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie_title', 'score', 'updated_at')
    list_filter = ('score',)
    search_fields = ('user__username', 'movie_title')


@admin.register(WatchlistItem)
class WatchlistItemAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie_title', 'added_at')
    search_fields = ('user__username', 'movie_title')


@admin.register(WatchHistory)
class WatchHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie_title', 'watched_at')
    search_fields = ('user__username', 'movie_title')


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('user', 'movie_title', 'score', 'like_count', 'created_at')
    search_fields = ('user__username', 'movie_title')


@admin.register(ReviewComment)
class ReviewCommentAdmin(admin.ModelAdmin):
    list_display = ('user', 'review', 'created_at')
    search_fields = ('user__username',)


@admin.register(MovieList)
class MovieListAdmin(admin.ModelAdmin):
    list_display = ('owner', 'title', 'is_public', 'updated_at')
    search_fields = ('owner__username', 'title')
    prepopulated_fields = {'slug': ('title',)}


@admin.register(MovieListItem)
class MovieListItemAdmin(admin.ModelAdmin):
    list_display = ('movie_list', 'movie_title', 'added_at')
