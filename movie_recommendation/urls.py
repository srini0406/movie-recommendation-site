from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from django.http import HttpResponse
from django.views.generic import TemplateView

def robots_txt(request):
    content = "User-agent: *\nAllow: /\nSitemap: https://movie-recommendation-site.onrender.com/sitemap.xml"
    return HttpResponse(content, content_type="text/plain")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('robots.txt', robots_txt),
    path('sitemap.xml', TemplateView.as_view(template_name='sitemap.xml', content_type='application/xml')),
    path('', include('recommender.urls')),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
