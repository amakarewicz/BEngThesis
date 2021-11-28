from django import forms

choices = [
    ('kmeans', 'KMeans'),
    ('hierarchical', 'Hierarchical'),
    ('dbscan', 'DBSCAN')
]


class CustomizeReport(forms.Form):
    algorithm = forms.ChoiceField(choices=choices, widget=forms.RadioSelect(attrs={'class': "algorithms_radio_select"}), label='Choose model')
