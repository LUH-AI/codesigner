from django import forms

from core.io import demo_datasets

from .models import Experiment
from .registry import MODELS, OPTIMIZERS


class NewExperimentForm(forms.Form):
    """Set up an experiment: name, model, optimizer, dataset, seed.

    A dataset comes from either the demo dropdown or an upload; exactly one is
    required. Creating an experiment does not run it — the metric to optimize
    and the number of trials are chosen per-run on the detail page. All metrics
    are always scored. Optimizer parameters use their defaults here (editing
    them is a later step).
    """

    name = forms.CharField(label="Experiment name", max_length=200)
    model_name = forms.ChoiceField(label="Model")
    optimizer_name = forms.ChoiceField(label="Optimizer")
    demo_dataset = forms.ChoiceField(label="Demo dataset", required=False)
    dataset_file = forms.FileField(label="…or upload a CSV (last column = target)", required=False)
    seed = forms.IntegerField(label="Seed (negative = random)", initial=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["model_name"].choices = [(k, k) for k in MODELS]
        self.fields["optimizer_name"].choices = [(k, k) for k in OPTIMIZERS]
        demos = demo_datasets()
        self.fields["demo_dataset"].choices = [("", "— none —")] + [(p, k) for k, p in demos.items()]

    def clean_name(self):
        name = self.cleaned_data["name"]
        if Experiment.objects.filter(name=name).exists():
            raise forms.ValidationError(f"An experiment named '{name}' already exists.")
        return name

    def clean(self):
        cleaned = super().clean()
        if not cleaned.get("demo_dataset") and not cleaned.get("dataset_file"):
            raise forms.ValidationError("Choose a demo dataset or upload a CSV file.")
        return cleaned
