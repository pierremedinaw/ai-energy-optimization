"""Pruebas del pipeline ML sin datos del mundo real."""

from datetime import date
import pandas as pd
import pytest

from energy_opt.forecast import add_solar_and_outages, forecast_day, make_demo_history


TARGET = date(2026, 10, 20)


def test_history_precedes_target():
    history = make_demo_history(TARGET, days=20)
    assert len(history) == 20 * 24
    assert history["timestamp"].max() < pd.Timestamp(TARGET)


def test_forecast_length_positive_and_metrics():
    forecast, metrics = forecast_day(TARGET, history_days=30)
    assert len(forecast) == 24
    assert (forecast["demand_kw"] > 0).all()
    assert metrics["validation_mae_kw"] >= 0
    assert metrics["naive_mae_kw"] >= 0


def test_solar_night_and_outage():
    forecast, _ = forecast_day(TARGET, history_days=30)
    enriched = add_solar_and_outages(forecast, 40, (18, 19))
    assert enriched.loc[0, "pv_kw"] == pytest.approx(0)
    assert enriched.loc[12, "pv_kw"] == pytest.approx(32)
    assert not enriched.loc[18, "grid_available"]
    assert enriched.loc[17, "grid_available"]
