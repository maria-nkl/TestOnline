from django.urls import path

from .views import ArticleListView, ArticleDetailView

urlpatterns = [
    path('', ArticleListView.as_view(), name='home'),
    path('articles/<str:slug>/', ArticleDetailView.as_view(), name='articles_detail'),
]