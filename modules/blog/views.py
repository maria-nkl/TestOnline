from django.shortcuts import render

from .models import Article


def articles_list(request):
    articles = Article.objects.all()
    return render(request, 'blog/articles_list.html', {'articles': articles})
