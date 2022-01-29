from django import forms

from application.models import Variable

algorithm_choices = [
    ('kmeans', 'K-Means'),
    ('hierarchical', 'Agglomerative'),
    ('dbscan', 'DBSCAN')
]

linkage_choices = [
    ('average', 'average'),
    ('complete', 'complete'),
    ('single', 'single')
]


class CustomizeReport(forms.Form):
    algorithm = forms.ChoiceField(choices=algorithm_choices,
                                  widget=forms.RadioSelect(attrs={'class': "radio_select"}),
                                  label='Choose model',
                                  initial='kmeans')
    variables = forms.ModelMultipleChoiceField(queryset=Variable.objects.all(),
                                               widget=forms.SelectMultiple(),
                                               label='Choose variables',
                                               initial=Variable.objects.all()) #CheckboxSelectMultiple
    n_clusters = forms.ChoiceField(choices=[(x, x) for x in range(2, 9)],
                                   label='Number of clusters',
                                   initial=4)
    linkage = forms.ChoiceField(choices=linkage_choices,
                                widget=forms.RadioSelect(attrs={'class': "radio_select"}),
                                label='Linkage',
                                initial='complete')

    eps = forms.DecimalField(
        label='Eps',
        widget=forms.TextInput(
            attrs={
                'step': '0.1',
                'type': 'range',
                'value': '3.4',
                'min': '2',
                'max': '11',
                'list': 'steplist',
                'width': '50%'

            }
        )
    )
    min_samples = forms.ChoiceField(choices=[(x, x) for x in range(2, 11)],
                                    label='Min samples',
                                    initial=7)

