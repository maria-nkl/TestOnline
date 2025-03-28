from django import forms
from django.core.validators import FileExtensionValidator
from .models import Article, Comment, ArticleFile


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result


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
            self.fields[field].widget.attrs.update({
                'class': 'form-control',
                'autocomplete': 'off'
            })

        self.fields['short_description'].widget.attrs.update({'class': 'form-control django_ckeditor_5'})
        self.fields['full_description'].widget.attrs.update({'class': 'form-control django_ckeditor_5'})
        self.fields['short_description'].required = False
        self.fields['full_description'].required = False
        self.fields['files'].widget.attrs.update({'class': 'form-control'})


class ArticleUpdateForm(ArticleCreateForm):
    delete_files = forms.ModelMultipleChoiceField(
        queryset=None,
        widget=forms.CheckboxSelectMultiple,
        required=False,
        label='Удалить файлы'
    )

    class Meta:
        model = Article
        fields = ArticleCreateForm.Meta.fields + ('updater', 'fixed')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance.pk:
            self.fields['delete_files'].queryset = self.instance.files.all()

        self.fields['fixed'].widget.attrs.update({'class': 'form-check-input'})
        self.fields['delete_files'].widget.attrs.update({'class': 'form-check-input'})


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
