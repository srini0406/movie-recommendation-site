from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
from django.utils.text import slugify
import uuid


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(blank=True, max_length=500)
    favorite_genres = models.JSONField(default=list, blank=True)
    avatar_url = models.URLField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username}'s profile"


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.get_or_create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    # Use get_or_create so this never crashes on users without a profile
    profile, _ = UserProfile.objects.get_or_create(user=instance)
    profile.save()


class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ratings')
    movie_title = models.CharField(max_length=255)
    score = models.PositiveSmallIntegerField()  # 1–10
    review = models.TextField(blank=True, max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'movie_title')
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.user.username} → {self.movie_title}: {self.score}/10"


class WatchlistItem(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watchlist')
    movie_title = models.CharField(max_length=255)
    added_at = models.DateTimeField(auto_now_add=True)
    imdb_id = models.CharField(max_length=20, blank=True)
    poster_url = models.URLField(blank=True)
    genres = models.CharField(max_length=255, blank=True)
    rating = models.CharField(max_length=20, blank=True)

    class Meta:
        unique_together = ('user', 'movie_title')
        ordering = ['-added_at']

    def __str__(self):
        return f"{self.user.username} → {self.movie_title}"


class WatchHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='watch_history')
    movie_title = models.CharField(max_length=255)
    watched_at = models.DateTimeField(auto_now_add=True)
    imdb_id = models.CharField(max_length=20, blank=True)
    poster_url = models.URLField(blank=True)
    genres = models.CharField(max_length=255, blank=True)
    rating = models.CharField(max_length=20, blank=True)

    class Meta:
        unique_together = ('user', 'movie_title')
        ordering = ['-watched_at']

    def __str__(self):
        return f"{self.user.username} watched {self.movie_title}"


# ---------------------------------------------------------------------------
# Phase 3 — Social & Engagement
# ---------------------------------------------------------------------------

class Review(models.Model):
    """Public text review for a movie, separate from private Rating."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='reviews')
    movie_title = models.CharField(max_length=255, db_index=True)
    body = models.TextField(max_length=2000)
    score = models.PositiveSmallIntegerField(null=True, blank=True)   # optional 1-10
    likes = models.ManyToManyField(User, related_name='liked_reviews', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ('user', 'movie_title')
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} reviewed {self.movie_title}"

    @property
    def like_count(self):
        return self.likes.count()


class ReviewComment(models.Model):
    """Threaded comment on a review."""
    review = models.ForeignKey(Review, on_delete=models.CASCADE, related_name='comments')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='review_comments')
    body = models.TextField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.user.username} on review #{self.review_id}"


class MovieList(models.Model):
    """Community-curated movie list (e.g. 'Weekend Watches', 'Oscar Winners')."""
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='movie_lists')
    title = models.CharField(max_length=150)
    description = models.TextField(blank=True, max_length=500)
    slug = models.SlugField(max_length=180, unique=True, blank=True)
    is_public = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def save(self, *args, **kwargs):
        if not self.slug:
            base = slugify(f"{self.owner.username}-{self.title}")
            self.slug = f"{base}-{uuid.uuid4().hex[:6]}"
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.owner.username}: {self.title}"


class MovieListItem(models.Model):
    movie_list = models.ForeignKey(MovieList, on_delete=models.CASCADE, related_name='items')
    movie_title = models.CharField(max_length=255)
    note = models.CharField(max_length=255, blank=True)
    poster_url = models.URLField(blank=True)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('movie_list', 'movie_title')
        ordering = ['added_at']

    def __str__(self):
        return f"{self.movie_list.title} → {self.movie_title}"
