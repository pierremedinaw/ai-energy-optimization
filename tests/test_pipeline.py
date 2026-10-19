"""Validación extremo a extremo con predicción y optimizador."""

from datetime import date
import numpy as np

from energy_opt.pipeline import run_demo
from energy_opt.scenario import Scenario


def test_end_to_end_run():
    actual, baseline, quality = run_demo(date(2026, 10, 20), Scenario())
    assert len(actual.timeseries) == 24
    assert np.isfinite(actual.metrics["objective_eur"])
    assert len(baseline.timeseries) == 24
    assert quality["test_hours"] > 0


def test_no_battery_scenario_matches_baseline():
    actual, baseline, _ = run_demo(date(2026, 10, 20), Scenario(
        battery_capacity_kwh=0, battery_power_kw=0,
    ))
    assert actual.metrics["objective_eur"] == baseline.metrics["objective_eur"]
