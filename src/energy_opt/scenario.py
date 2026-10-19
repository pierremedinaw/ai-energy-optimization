"""Parámetros y validaciones del escenario energético."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    """Supuestos genéricos del sistema, no curvas certificadas de fabricantes.

    Potencias en kW, energías en kWh y costes en EUR/kWh.
    La discretización del MVP es horaria.
    """

    pv_peak_kw: float = 60.0
    battery_capacity_kwh: float = 90.0
    battery_power_kw: float = 40.0
    soc_min: float = 0.10
    soc_initial: float = 0.50
    soc_max: float = 0.95
    eta_charge: float = 0.94
    eta_discharge: float = 0.94
    grid_max_kw: float = 160.0
    gas_max_kw: float = 50.0
    diesel_max_kw: float = 75.0
    grid_cost_eur_kwh: float = 0.22
    gas_cost_eur_kwh: float = 0.19
    diesel_cost_eur_kwh: float = 0.34
    battery_wear_eur_kwh: float = 0.015
    unserved_penalty_eur_kwh: float = 100.0
    grid_emission_kg_kwh: float = 0.25
    gas_emission_kg_kwh: float = 0.50
    diesel_emission_kg_kwh: float = 0.74
    carbon_price_eur_tonne: float = 0.0
    outage_hours: tuple[int, ...] = (18, 19, 20)
    time_step_hours: float = 1.0

    def __post_init__(self) -> None:
        # Comprueba límites físicos y unidades de manera anticipada.
        nonnegative = (
            "pv_peak_kw", "battery_capacity_kwh", "battery_power_kw", "grid_max_kw",
            "gas_max_kw", "diesel_max_kw", "grid_cost_eur_kwh", "gas_cost_eur_kwh",
            "diesel_cost_eur_kwh", "battery_wear_eur_kwh", "carbon_price_eur_tonne",
            "grid_emission_kg_kwh", "gas_emission_kg_kwh", "diesel_emission_kg_kwh",
        )
        for name in nonnegative:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} no puede ser negativo")
        if not 0 <= self.soc_min <= self.soc_initial <= self.soc_max <= 1:
            raise ValueError("Debe cumplirse 0 <= soc_min <= soc_initial <= soc_max <= 1")
        if not (0 < self.eta_charge <= 1 and 0 < self.eta_discharge <= 1):
            raise ValueError("Las eficiencias deben estar en el intervalo (0, 1]")
        if self.unserved_penalty_eur_kwh <= 0 or self.time_step_hours <= 0:
            raise ValueError("Penalización y paso temporal deben ser positivos")
        if any(hour not in range(24) for hour in self.outage_hours):
            raise ValueError("Las horas de corte deben estar entre 0 y 23")
