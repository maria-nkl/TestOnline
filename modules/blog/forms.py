from django import forms
from django.core.validators import FileExtensionValidator
from .models import Article, Comment, ArticleFile


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput(attrs={'multiple': True}))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        if data is None:
            return []
        if isinstance(data, (list, tuple)):
            return [super(MultipleFileField, self).clean(d) for d in data]
        return [super(MultipleFileField, self).clean(data)]


class ArticleCreateForm(forms.ModelForm):
    files = MultipleFileField(
        label='Файлы (JPG/PDF)',
        required=False,
        validators=[FileExtensionValidator(allowed_extensions=('jpg', 'jpeg', 'pdf'))]
    )

    class Meta:
        model = Article
        fields = ('title', 'slug', 'category', 'short_description', 'full_description', 'thumbnail', 'status')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields:
            if field != 'files':
                self.fields[field].widget.attrs.update({
                    'class': 'form-control',
                    'autocomplete': 'off'
                })

        self.fields['short_description'].widget.attrs.update({'class': 'form-control django_ckeditor_5'})
        self.fields['full_description'].widget.attrs.update({'class': 'form-control django_ckeditor_5'})
        self.fields['files'].widget.attrs.update({'class': 'form-control-file'})


class ArticleUpdateForm(ArticleCreateForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'fixed' in self.fields:  # Проверяем наличие поля перед обновлением атрибутов
            self.fields['fixed'].widget.attrs.update({'class': 'form-check-input'})

    class Meta(ArticleCreateForm.Meta):
        fields = ArticleCreateForm.Meta.fields + ('fixed', 'updater')


class CommentCreateForm(forms.ModelForm):
    parent = forms.IntegerField(widget=forms.HiddenInput, required=False)
    content = forms.CharField(label='', widget=forms.Textarea(attrs={
        'cols': 30,
        'rows': 5,
        'placeholder': 'Комментарий',
        'class': 'form-control'
    }))

    class Meta:
        model = Comment
        fields = ('content',)
