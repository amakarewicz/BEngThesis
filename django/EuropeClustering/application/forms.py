from django import forms

from application.models import Variable

algorithm_choices = [
    ('kmeans', 'KMeans'),
    ('hierarchical', 'Hierarchical'),
    ('dbscan', 'DBSCAN')
]
variables_choices = Variable.objects.all()


class CustomizeReport(forms.Form):
    algorithm = forms.ChoiceField(choices=algorithm_choices, widget=forms.RadioSelect(attrs={'class': "algorithms_radio_select"}), label='Choose model')
    variables = forms.ModelMultipleChoiceField(queryset=variables_choices, widget=forms.SelectMultiple(), label='Choose variables') #CheckboxSelectMultiple
