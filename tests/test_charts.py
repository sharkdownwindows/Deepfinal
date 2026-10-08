import csv

import pytest

from personalized_t2i.reporting.charts import (
    _plot_metric,
    load_aggregate_metrics,
)


def test_aggregate_metrics_require_generation_mode(tmp_path):
    path = tmp_path / "metrics_aggregate.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["run_id", "concept_id", "rank", "data_size", "dino_mean", "dino_sd", "clip_mean", "clip_sd"],
        )
        writer.writeheader()

    with pytest.raises(ValueError, match="generation_mode"):
        load_aggregate_metrics(path)


def test_chart_keeps_generation_modes_as_separate_series(tmp_path, monkeypatch):
    from personalized_t2i.reporting import charts

    plotted = []

    class Axis:
        def plot(self, x_values, y_values, **kwargs):
            plotted.append((x_values, y_values, kwargs["label"]))

        def set_xlabel(self, _value):
            pass

        def set_ylabel(self, _value):
            pass

        def set_title(self, _value):
            pass

        def grid(self, *_args, **_kwargs):
            pass

        def legend(self, **_kwargs):
            pass

    class Figure:
        def text(self, *_args, **_kwargs):
            pass

        def tight_layout(self, **_kwargs):
            pass

        def savefig(self, path, **_kwargs):
            path.write_bytes(b"chart")

    monkeypatch.setattr(charts.plt, "subplots", lambda **_kwargs: (Figure(), Axis()))
    monkeypatch.setattr(charts.plt, "close", lambda _figure: None)

    rows = [
        {
            "run_id": "dog_n5_r16",
            "generation_mode": mode,
            "concept_id": "dog",
            "data_size": "5",
            "dino_mean": score,
        }
        for mode, score in (("base", "0.2"), ("adapter", "0.8"))
    ]
    output_path = tmp_path / "metric.png"

    _plot_metric(
        rows,
        x_column="data_size",
        x_label="Training data size",
        metric_column="dino_mean",
        metric_label="DINO",
        title="DINO by mode",
        output_path=output_path,
    )

    assert [plot[2] for plot in plotted] == ["dog (adapter)", "dog (base)"]
    assert output_path.is_file()
