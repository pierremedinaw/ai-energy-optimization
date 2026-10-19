"""Pruebas deterministas de los invariantes eléctricos y de despacho."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from energy_opt.dispatch import optimize_dispatch
from energy_opt.scenario import Scenario


def hourly(load=40.0, pv=0.0, hours=5, outage=()):
    timestamps = pd.date_range("2026-10-20", periods=hours, freq="h")
    return pd.DataFrame({"timestamp": timestamps,
                         "demand_kw": [load] * hours,
                         "pv_kw": [pv] * hours,
                         "grid_available": [i not in outage for i in range(hours)]})


def test_grid_supplies_demand_when_cheapest():
    scenario = Scenario(battery_capacity_kwh=0, battery_power_kw=0,
                        gas_max_kw=0, diesel_max_kw=0, outage_hours=())
    result = optimize_dispatch(hourly(), scenario)
    assert result.metrics["unserved_kwh"] == pytest.approx(0, abs=1e-7)
    assert result.metrics["grid_kwh"] == pytest.approx(200)


def test_grid_is_zero_during_outage():
    result = optimize_dispatch(hourly(outage=(1, 2)), Scenario())
    assert (result.timeseries.loc[[1, 2], "grid_kw"] < 1e-7).all()


def test_diesel_only_produces_when_gas_insufficient():
    scenario = Scenario(battery_capacity_kwh=0, battery_power_kw=0,
                        gas_max_kw=20, diesel_max_kw=40, outage_hours=(0, 1, 2, 3, 4))
    result = optimize_dispatch(hourly(load=45, outage=tuple(range(5))), scenario)
    assert np.allclose(result.timeseries["gas_kw"], 20)
    assert np.allclose(result.timeseries["diesel_kw"], 25)


def test_unserved_load_when_all_sources_unavailable():
    scenario = Scenario(battery_capacity_kwh=0, battery_power_kw=0,
                        gas_max_kw=0, diesel_max_kw=0)
    result = optimize_dispatch(hourly(load=50, outage=tuple(range(5))), scenario)
    assert result.metrics["unserved_kwh"] == pytest.approx(250)
    assert result.metrics["served_pct"] == pytest.approx(0)


def test_battery_bounds_and_terminal_reserve():
    scenario = Scenario(battery_capacity_kwh=60, battery_power_kw=20,
                        gas_max_kw=0, diesel_max_kw=0)
    result = optimize_dispatch(hourly(load=30, pv=5, hours=24, outage=(18, 19)), scenario)
    soc = result.timeseries["soc_kwh"]
    assert (soc >= scenario.soc_min * 60 - 1e-6).all()
    assert (soc <= scenario.soc_max * 60 + 1e-6).all()
    assert soc.iloc[-1] >= scenario.soc_initial * 60 - 1e-6


def test_power_balance_each_hour():
    result = optimize_dispatch(hourly(load=65, pv=12, hours=24, outage=(8, 9, 10)), Scenario())
    assert np.max(np.abs(result.timeseries["supply_balance_kw"])) < 1e-6


def test_unserved_reduced_by_backup():
    load = hourly(load=50, hours=5, outage=tuple(range(5)))
    without = optimize_dispatch(load, Scenario(battery_capacity_kwh=0, battery_power_kw=0,
                                              gas_max_kw=0, diesel_max_kw=0))
    with_gas = optimize_dispatch(load, Scenario(battery_capacity_kwh=0, battery_power_kw=0,
                                               gas_max_kw=50, diesel_max_kw=0))
    assert with_gas.metrics["unserved_kwh"] < without.metrics["unserved_kwh"]


def test_carbon_price_changes_preference():
    load = hourly(load=30, hours=3)
    base = Scenario(battery_capacity_kwh=0, battery_power_kw=0, gas_max_kw=30,
                    diesel_max_kw=0, outage_hours=(), grid_cost_eur_kwh=0.18,
                    gas_cost_eur_kwh=0.16, grid_emission_kg_kwh=0.10,
                    gas_emission_kg_kwh=0.70)
    ordinary = optimize_dispatch(load, base)
    penalized = optimize_dispatch(load, replace(base, carbon_price_eur_tonne=300))
    assert ordinary.metrics["gas_kwh"] > 0
    assert penalized.metrics["grid_kwh"] > ordinary.metrics["grid_kwh"]


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="Columnas faltantes"):
        optimize_dispatch(hourly().drop(columns="pv_kw"), Scenario())


def test_invalid_soc_rejected():
    with pytest.raises(ValueError, match="soc_min"):
        Scenario(soc_min=0.7, soc_initial=0.5)


def test_invalid_outage_hour_rejected():
    with pytest.raises(ValueError, match="horas de corte"):
        Scenario(outage_hours=(24,))


def test_negative_demand_rejected():
    with pytest.raises(ValueError, match="demand_kw"):
        optimize_dispatch(hourly(load=-1), Scenario())


def test_pv_curtailed_if_exceeds_load_without_battery():
    scenario = Scenario(battery_capacity_kwh=0, battery_power_kw=0,
                        gas_max_kw=0, diesel_max_kw=0, grid_max_kw=0)
    result = optimize_dispatch(hourly(load=10, pv=50), scenario)
    assert result.metrics["curtailed_pv_kwh"] == pytest.approx(200)
    assert result.metrics["unserved_kwh"] == pytest.approx(0)
