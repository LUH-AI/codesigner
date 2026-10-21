"""The configuration cube's prediction layer: what the surrogate makes of the
whole search space, drawn under the trials.

One request per metric (`surrogate_prediction`) returns an even sample of the
space with its predictions and its place in every view, so the page switches
view, dimensions and axes without asking again. Uses the test2.ihpo fixture
(Random Forest, Random Search, 30 trials).
"""

import json

from django.urls import reverse

from core import io
from experiments.services import snapshot as adapter

from tests.conftest import FIXTURES_DIR


def _experiment(fixture="test2.ihpo"):
    """An experiment read from a stored .ihpo, as the import page would create it."""
    return adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / fixture).read_bytes()))


def _layer(client, exp, metric="accuracy"):
    return client.get(reverse("ui:surrogate_prediction", args=[exp.pk]) + f"?metric={metric}")


def test_one_answer_carries_every_view(client):
    """The sample comes back with its hyperparameter values and its place in
    both projections, at every dimensionality, outlines included.

    Expect: one row of values, one mean and one spread per sample; PCA and PLS
    coordinates for every sample; an outline for one, two and three
    components; and the best observed score for expected improvement.
    """
    data = _layer(client, _experiment()).json()

    n = len(data["values"])
    assert n > 0 and len(data["mu"]) == n and len(data["sigma"]) == n
    assert len(data["values"][0]) == len(data["hp_names"])
    for method in ("pca", "pls"):
        projection = data["projections"][method]
        assert len(projection["coordinates"]) == n
        assert set(projection["outlines"]) == {"1", "2", "3"}
    assert data["eta"] is not None


def test_a_search_with_no_model_offers_no_acquisition(client):
    """Random search fits no model of the objective, so it has no acquisition
    function: the answer says so, and the page takes the option away."""
    assert _layer(client, _experiment()).json()["acquisition"] is False


def test_rejects_an_unknown_metric(client):
    assert _layer(client, _experiment(), metric="not-a-metric").status_code == 400


def test_the_cube_offers_the_layer_and_every_dimensionality(client):
    """A function selector on the cube, at the acquisition as the page is
    served with None beside it, and a 1D option for the projections beside 2D
    and 3D."""
    html = client.get(reverse("ui:experiment_detail", args=[_experiment().pk])).content.decode()
    layer = html.split('id="cube-layer"', 1)[1].split("</select>", 1)[0]
    dimensions = html.split('id="cube-dimensions"', 1)[1].split("</select>", 1)[0]

    assert 'value="acquisition" selected' in layer
    assert 'value="prediction"' in layer and 'value=""' in layer
    for value in ("1", "2", "3"):
        assert f'value="{value}"' in dimensions


def test_the_layer_opens_on_its_default_whatever_a_reload_restored(client):
    """A browser restores a select across a reload. The page puts the layer
    back on the acquisition as a field — or on None where the acquisition
    figure's settings say to wait to be asked — rather than open on whatever
    was last looked at. Asserted on the page script."""
    html = client.get(reverse("ui:experiment_detail", args=[_experiment().pk])).content.decode()
    assert ('layerSelect.value = autocompute.acquisition_slice === false ? "" : "acquisition";'
            in html)
    assert 'layerStyleSelect.value = "field";' in html


def test_the_layer_waits_for_a_run_to_end(client):
    """While a run adds trials every poll would mean a new sample, so the
    layer's selectors are served disabled and the page is told a run is going,
    which keeps every computed figure from fetching on its own.

    Setup: test2.ihpo with a run marked as running.
    Expect: both selectors disabled and the page's running flag set.
    """
    from experiments.models import Run

    exp = _experiment()
    Run.objects.create(experiment=exp, stopping={"max_trials": 3}, primary_metric="accuracy",
                       status="running")
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    for select in ("cube-layer", "cube-layer-style"):
        tag = html.split(f'id="{select}"', 1)[1].split(">", 1)[0]
        assert "disabled" in tag, select
    assert "window.codesignerComputed.running = true;" in html


def test_a_finished_run_leaves_the_layer_enabled(client):
    html = client.get(reverse("ui:experiment_detail", args=[_experiment().pk])).content.decode()
    tag = html.split('id="cube-layer"', 1)[1].split(">", 1)[0]
    assert "disabled" not in tag
    assert "window.codesignerComputed.running = true;" not in html


def test_a_stated_prior_comes_with_the_layer_as_a_weight(client):
    """Each stated prior arrives as its density over the acquisition figure's
    grid and the exponent its decay has reached, with every sample's vectorized
    position beside it — everything the page needs to weight the acquisition,
    and to keep weighting it as the prior is edited, without a second request.

    Setup: test2.ihpo with a Normal prior on max_features at 0.3.
    Expect: the prior keyed by its hyperparameter, its density peaking at the
    grid point nearest 0.3, a positive exponent, and one vector per sample.
    """
    exp = _experiment()
    exp.data.priors = {"max_features": {"kind": "normal",
                                        "params": {"mu": 0.3, "sigma": 0.1}}}
    exp.data.save(update_fields=["priors"])

    data = _layer(client, exp).json()

    prior = data["priors"]["max_features"]
    peak = prior["positions"][prior["density"].index(max(prior["density"]))]
    assert abs(peak - 0.3) <= 0.01
    assert prior["exponent"] > 0 and prior["categorical"] is False
    assert len(data["vectors"]) == len(data["values"])


def test_no_prior_means_no_weight(client):
    assert _layer(client, _experiment()).json()["priors"] == {}


def test_a_search_with_no_model_proposes_nothing(client):
    """Random search cannot be asked what it would run next: the endpoint says
    so rather than failing, and the cube offers no Next Trials for it."""
    exp = _experiment()
    data = client.get(reverse("ui:next_trials", args=[exp.pk])).json()
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    assert data["candidates"] is None
    assert 'id="cube-next-trials"' not in html and 'id="cube-requery"' not in html


def test_a_model_based_search_offers_next_trials_on_the_cube(client):
    """An optimizer that can be asked gets the cube's Next Trials checkbox with
    a Requery button, served unticked and disabled: nothing is asked for until
    the box is ticked."""
    exp = _experiment()
    exp.data.optimizer_name = "SMAC"
    exp.data.save(update_fields=["optimizer_name"])
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()

    box = html.split('id="cube-next-trials"', 1)[1].split(">", 1)[0]
    button = html.split('id="cube-requery"', 1)[1].split(">", 1)[0]
    assert "data-next-trials" in box and "checked" not in box
    assert "data-requery" in button and "disabled" in button


def test_both_figures_carry_one_pair_of_next_trials_controls(client):
    """The acquisition figure and the cube ask the same question, so their
    controls are the same controls: both marked for the one script that keeps
    them in step, both asking the same endpoint.

    Setup: test2.ihpo as a SMAC experiment.
    Expect: two checkboxes and two Requery buttons, both checkboxes pointing at
    next_trials, and that script loaded once.
    """
    exp = _experiment()
    exp.data.optimizer_name = "SMAC"
    exp.data.save(update_fields=["optimizer_name"])
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    url = reverse("ui:next_trials", args=[exp.pk])

    assert html.count("data-next-trials ") == 2
    assert html.count("data-requery ") == 2
    assert html.count(f'data-url="{url}"') == 2
    assert html.count("ui/next_trials") == 1


def test_proposals_can_be_anchored_at_a_hyperparameter(client, monkeypatch):
    """The acquisition figure asks with its hyperparameter, whose prior the
    trial budget is anchored at; the cube asks without one."""
    from ui.views import priors

    anchors = []
    monkeypatch.setattr(priors, "_next_trials",
                        lambda exp, anchor_hp=None: anchors.append(anchor_hp))
    exp = _experiment()
    url = reverse("ui:next_trials", args=[exp.pk])
    client.get(url + "?hp=max_features")
    client.get(url)

    assert anchors == ["max_features", None]


def test_the_projection_payload_carries_its_linear_map(client):
    """The cube's own projection views carry the map that places a proposed
    configuration, so candidates can be drawn whether or not a layer is on."""
    exp = _experiment()
    plots = client.get(reverse("ui:metric_figures", args=[exp.pk]) + "?metric=accuracy").json()
    for view in ("pca", "pls"):
        meta = plots["configuration_cube"][view]["layout"]["meta"]
        assert set(meta["map"]) == {"columns", "matrix", "offset"}


def test_the_two_figures_share_one_fit_and_one_sample(client, monkeypatch):
    """The acquisition figure and the configuration cube read the same sample of
    the space through the same fitted model, so whichever asks second costs no
    fit and no prediction.

    Setup: count surrogate fits and samplings.
    Action: ask for the acquisition slice, then the cube's prediction layer, for
    the same metric.
    Expect: one fit and one sample between them.
    """
    from core import analytics
    from core.optimizers.base import BaseOptimizer
    from ui.views import priors

    priors._SLICE_SURROGATES.clear()
    fits, samples = [], []
    fit, sample = BaseOptimizer.slice_surrogate, analytics.sample_prediction_space

    def counted_fit(self, *args, **kwargs):
        fits.append(1)
        return fit(self, *args, **kwargs)

    def counted_sample(*args, **kwargs):
        samples.append(1)
        return sample(*args, **kwargs)

    monkeypatch.setattr(BaseOptimizer, "slice_surrogate", counted_fit)
    monkeypatch.setattr(analytics, "sample_prediction_space", counted_sample)

    exp = _experiment()
    slice_url = reverse("ui:acquisition_slice", args=[exp.pk]) + "?metric=accuracy&hp=max_features"
    assert client.get(slice_url).status_code == 200
    assert _layer(client, exp).status_code == 200

    assert len(fits) == 1
    assert len(samples) == 1


def test_the_acquisition_cloud_is_the_shared_sample(client):
    """The acquisition figure's cloud is the cube's sample, read along one
    hyperparameter: one point per sample where that hyperparameter is active."""
    from ui.views import priors

    priors._SLICE_SURROGATES.clear()
    exp = _experiment()
    sample = _layer(client, exp).json()
    column = sample["hp_names"].index("max_features")
    active = sum(1 for row in sample["vectors"] if row[column] is not None)

    slice_url = reverse("ui:acquisition_slice", args=[exp.pk]) + "?metric=accuracy&hp=max_features"
    cloud = client.get(slice_url).json()["meta"]["cloud"]

    assert len(cloud["positions"]) == active


def test_a_proposal_carries_its_place_on_every_axis():
    """Each proposed configuration says where it sits on every hyperparameter's
    vectorized axis, so either figure can place it on the axis it is drawing
    without asking the optimizer again."""
    from ConfigSpace import Configuration

    from core.models import RandomForestModel
    from ui.views.priors import _candidate

    cs = RandomForestModel().get_config_space(seed=0)
    config = dict(cs.get_default_configuration())
    candidate = _candidate(1, config, "Local Search", None, cs)

    expected = Configuration(cs, values=config).get_array()
    assert list(candidate["positions"]) == list(cs.keys())
    for name, value in zip(cs.keys(), expected):
        assert abs(candidate["positions"][name] - value) < 1e-12


def test_a_trial_in_the_cube_says_its_configuration_on_hover(client):
    """In a projection a point's coordinates name none of its values, so each
    trial's hover carries its whole configuration and its score."""
    exp = _experiment()
    plots = client.get(reverse("ui:metric_figures", args=[exp.pk]) + "?metric=accuracy").json()
    for view in ("axes", "pca", "pls"):
        trace = plots["configuration_cube"][view]["data"][0]
        first = trace["hovertext"][0]
        assert first.startswith("Trial ") and "max_depth = " in first and "Accuracy" in first, view


def test_the_layer_is_a_field_or_a_point_cloud(client):
    """A selector under the function's, read "as", opening on the field. It
    has no None of its own: with no function it has nothing to say, and the
    page disables it — asserted on the page script."""
    html = client.get(reverse("ui:experiment_detail", args=[_experiment().pk])).content.decode()
    style = html.split('id="cube-layer-style"', 1)[1].split("</select>", 1)[0]

    assert 'value="field" selected' in style
    assert 'value="points"' in style and 'value="none"' not in style
    assert "layerStyleSelect.disabled = !currentLayer();" in html


def test_the_cube_controls_read_as_columns(client):
    """Method above "in" and its dimensions, function above "as" and its style,
    Next Trials above Requery — in that order in a grid filled column by
    column, so each pair shares a column.

    Setup: test2.ihpo as a SMAC experiment, so Next Trials is offered.
    Expect: the pairs in order in the cube's head, with "in" and "as" between
    each pair's halves, and the method relabelled HP Space.
    """
    exp = _experiment()
    exp.data.optimizer_name = "SMAC"
    exp.data.save(update_fields=["optimizer_name"])
    html = client.get(reverse("ui:experiment_detail", args=[exp.pk])).content.decode()
    head = html[html.index('data-figure="configuration_cube"'):]
    head = head[:head.index('id="prediction-warning"')]

    order = ["view-select-configuration_cube", "cube-dimensions-row", "cube-dimensions",
             "cube-layer", "cube-layer-style", "cube-next-trials", "cube-requery"]
    at = [head.index(f'id="{name}"') for name in order]
    assert at == sorted(at)
    between = lambda a, b: head[head.index(f'id="{a}"'):head.index(f'id="{b}"')]
    assert ">in<" in between("cube-dimensions-row", "cube-dimensions")
    assert ">as<" in between("cube-layer", "cube-layer-style")
    assert 'class="selectors stacked"' in head
    assert ">HP Space<" in head


def test_the_layer_says_which_acquisition_function_the_run_used(client):
    """The page computes the acquisition from the sample's mean and spread, so
    it is told which function and margin the run maximized: SMAC's own setting,
    and expected improvement for an optimizer that names none.

    Setup: test.ihpo (SMAC) as stored, then with probability of improvement
    and a margin.
    Expect: each answer names the run's function and margin.
    """
    from experiments.services import snapshot as adapter

    exp = adapter.experiment_from_snapshot(io.parse((FIXTURES_DIR / "test.ihpo").read_bytes()))
    assert _layer(client, exp).json()["acquisition_function"] == {"kind": "ei", "xi": 0.0}

    exp.data.optimizer_params = dict(exp.data.optimizer_params or {},
                                     acquisition="pi", acquisition_xi=0.01)
    exp.data.save(update_fields=["optimizer_params"])
    assert _layer(client, exp).json()["acquisition_function"] == {"kind": "pi", "xi": 0.01}


def test_random_search_falls_back_to_expected_improvement(client):
    assert _layer(client, _experiment()).json()["acquisition_function"] == {"kind": "ei", "xi": 0.0}


def test_the_cube_opens_the_same_way_every_time(client):
    """PLS in two dimensions with no axes picked, whatever a reload restored —
    and, with the layer opening on the acquisition as a field and Next Trials
    served unticked, that is the whole of its opening state. Asserted on the
    page script."""
    html = client.get(reverse("ui:experiment_detail", args=[_experiment().pk])).content.decode()

    assert 'cubeMethodSelect.value = "pls";' in html
    assert 'if (cubeDimensions) cubeDimensions.value = "2";' in html
    assert 'cubeAxisSelect[axis].value = "";' in html


def test_a_level_the_3d_field_never_reaches_is_neither_drawn_nor_marked(client):
    """Near the legal region's edge the 3D field eases away to close its shells,
    which can take it under a level the clipped field reached. Such a level
    would be a shell with nothing in it and a mark on the bar for a surface
    nobody can see, so the levels are checked against the eased field, and the
    bar is marked after. Asserted on the page script."""
    html = client.get(reverse("ui:experiment_detail", args=[_experiment().pk])).content.decode()
    traces = html.split("function layerTraces(", 1)[1].split("\n            }\n", 1)[0]

    assert "return vs.some(function (v)" in traces
    assert traces.index("return vs.some(function (v)") < traces.index("markLevels(")
