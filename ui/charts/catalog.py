"""The charts an experiment page can show.

A **chart** is one analytics panel on an experiment page. This list is what
defines them: everything else derives from it — the visibility settings and
their checkboxes on the default-experiment-settings page, the panels the detail
page renders, and the ids the page script looks them up by.

Adding a chart is two steps: an entry here, and the matching partial at
`ui/templates/ui/charts/<key>.html`. Nothing else needs to learn about it.
"""

from dataclasses import dataclass

from django.utils.translation import gettext_lazy as _


@dataclass(frozen=True)
class Chart:
    """One analytics panel: its stable key, its heading, and how it lays out."""

    key: str
    label: str
    #: "grid" charts tile two-up in reading order; "wide" ones span the page
    #: below them (a table with a column per metric needs the full width).
    layout: str = "grid"

    @property
    def template(self) -> str:
        """The partial that renders this chart."""
        return f"ui/charts/{self.key}.html"

    @property
    def setting_key(self) -> str:
        """The experiment-settings key controlling whether it is shown."""
        return f"show_{self.key}"

    @property
    def dom_id(self) -> str:
        """The id of the element the page script draws into."""
        return f"chart-{self.key}"


# Order is the order they appear on the page.
CHARTS = (
    Chart("best_configuration", _("Best configuration")),
    Chart("selected_configuration", _("Selected configuration")),
    # Keeps the existing capitalization of these two headings — they are already
    # translated, and rewording them would orphan the de/es entries.
    Chart("hyperparameter_importance", _("Hyperparameter importance (HyperSHAP)")),
    Chart("incumbent_performance", _("Performance of Incumbent")),
    Chart("trial_duration", _("Trial duration")),
    Chart("error_over_time", _("Error over time")),
    Chart("trials", _("Trials"), layout="wide"),
)

CHARTS_BY_KEY = {chart.key: chart for chart in CHARTS}
