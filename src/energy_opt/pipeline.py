"""Orquesta pronóstico, solar sintético y despacho horario."""

from dataclasses import replace
from datetime import date

from energy_opt.dispatch import DispatchResult, optimize_dispatch
from energy_opt.forecast import add_solar_and_outages, forecast_day
from energy_opt.scenario import Scenario


def run_demo(target_day: date, scenario: Scenario) -> tuple[DispatchResult, DispatchResult, dict[str, float]]:
    """Calcula los escenarios con y sin BESS bajo idéntica demanda y recursos.

    El caso sin BESS tiene capacidad y potencia cero; la comparación NO
    representa ahorro verificable, pues parte de datos y costes sintéticos.
    """
    demand_forecast, quality = forecast_day(target_day)
    profile = add_solar_and_outages(demand_forecast, scenario.pv_peak_kw, scenario.outage_hours)
    optimized = optimize_dispatch(profile, scenario)
    baseline = optimize_dispatch(profile, replace(scenario, battery_capacity_kwh=0.0,
                                                  battery_power_kw=0.0))
    return optimized, baseline, quality
