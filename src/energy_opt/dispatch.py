"""Despacho lineal multiperiodo de red, fotovoltaica, BESS, gas y diésel.

El MVP aproxima generadores como unidades de potencia continua regulable:
no modela arranques, potencia mínima, tiempos de rampa ni eficiencia parcial.
No debe usarse para controlar equipamiento físico.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from energy_opt.scenario import Scenario

# Variables para cada intervalo: potencias en kW y energía almacenada en kWh.
GRID, GAS, DIESEL, CHARGE, DISCHARGE, CURTAIL, UNSERVED, SOC = range(8)
NUM_VARS = 8


@dataclass
class DispatchResult:
    """Serie horaria y métricas del escenario optimizado."""

    timeseries: pd.DataFrame
    metrics: dict[str, float]


def optimize_dispatch(profile: pd.DataFrame, scenario: Scenario) -> DispatchResult:
    """Resuelve un programa lineal con conservación de energía y límites físicos.

    Prohíbe la importación de red durante los cortes y mantiene al finalizar
    al menos el SOC inicial; esto impide atribuir ahorro ficticio al SOC inicial.
    """
    required = {"timestamp", "demand_kw", "pv_kw", "grid_available"}
    if not required.issubset(profile.columns):
        raise ValueError(f"Columnas faltantes: {sorted(required - set(profile.columns))}")
    if profile.empty:
        raise ValueError("El perfil no puede estar vacío")
    frame = profile.copy().reset_index(drop=True)
    for column in ("demand_kw", "pv_kw"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy()).all() or (frame[column] < 0).any():
            raise ValueError(f"{column} debe tener valores finitos no negativos")
    if not frame["grid_available"].map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise ValueError("grid_available debe contener solo booleanos")

    n = len(frame)
    dt = scenario.time_step_hours
    cap = scenario.battery_capacity_kwh
    initial_soc = cap * scenario.soc_initial
    lower_soc = cap * scenario.soc_min
    upper_soc = cap * scenario.soc_max
    carbon_eur_kg = scenario.carbon_price_eur_tonne / 1000.0

    # Se minimiza coste operativo + coste ambiental + penalización por interrupción.
    objective = np.zeros(n * NUM_VARS)
    bounds: list[tuple[float, float | None]] = []
    for t, row in frame.iterrows():
        base = t * NUM_VARS
        objective[base + GRID] = dt * (
            scenario.grid_cost_eur_kwh + carbon_eur_kg * scenario.grid_emission_kg_kwh)
        objective[base + GAS] = dt * (
            scenario.gas_cost_eur_kwh + carbon_eur_kg * scenario.gas_emission_kg_kwh)
        objective[base + DIESEL] = dt * (
            scenario.diesel_cost_eur_kwh + carbon_eur_kg * scenario.diesel_emission_kg_kwh)
        objective[base + CHARGE] = dt * scenario.battery_wear_eur_kwh
        objective[base + DISCHARGE] = dt * scenario.battery_wear_eur_kwh
        objective[base + UNSERVED] = dt * scenario.unserved_penalty_eur_kwh
        soc_floor = max(lower_soc, initial_soc) if t == n - 1 else lower_soc
        bounds.extend([
            (0, scenario.grid_max_kw if row["grid_available"] else 0),
            (0, scenario.gas_max_kw),
            (0, scenario.diesel_max_kw),
            (0, scenario.battery_power_kw if cap > 0 else 0),
            (0, scenario.battery_power_kw if cap > 0 else 0),
            (0, float(row["pv_kw"])),
            (0, float(row["demand_kw"])),
            (soc_floor, upper_soc),
        ])

    equality = lil_matrix((n * 2, n * NUM_VARS), dtype=float)
    rhs = np.zeros(n * 2, dtype=float)
    for t, row in frame.iterrows():
        base = t * NUM_VARS
        # Balance de potencia: PV útil + fuentes + descarga + no servido = demanda + carga.
        for offset, value in (
            (GRID, 1), (GAS, 1), (DIESEL, 1), (DISCHARGE, 1),
            (UNSERVED, 1), (CHARGE, -1), (CURTAIL, -1),
        ):
            equality[t, base + offset] = value
        rhs[t] = float(row["demand_kw"] - row["pv_kw"])

        # SOC(t+1) = SOC(t) + eta_c*P_charge*dt - P_discharge*dt/eta_d.
        equality[n + t, base + SOC] = 1
        equality[n + t, base + CHARGE] = -scenario.eta_charge * dt
        equality[n + t, base + DISCHARGE] = dt / scenario.eta_discharge
        if t == 0:
            rhs[n + t] = initial_soc
        else:
            equality[n + t, (t - 1) * NUM_VARS + SOC] = -1

    solution = linprog(objective, A_eq=equality.tocsr(), b_eq=rhs,
                       bounds=bounds, method="highs")
    if not solution.success or solution.x is None:
        raise RuntimeError(f"La optimización no convergió: {solution.message}")

    values = solution.x.reshape((n, NUM_VARS))
    output = frame.copy()
    names = {
        GRID: "grid_kw", GAS: "gas_kw", DIESEL: "diesel_kw",
        CHARGE: "charge_kw", DISCHARGE: "discharge_kw",
        CURTAIL: "curtailed_pv_kw", UNSERVED: "unserved_kw", SOC: "soc_kwh",
    }
    for index, name in names.items():
        # Se limpia solamente ruido numérico del solver; no cambia resultados útiles.
        output[name] = np.where(np.abs(values[:, index]) < 1e-7, 0, values[:, index])
    output["pv_used_kw"] = output["pv_kw"] - output["curtailed_pv_kw"]
    output["soc_pct"] = (output["soc_kwh"] / cap * 100) if cap > 0 else 0.0
    output["supply_balance_kw"] = (
        output["grid_kw"] + output["gas_kw"] + output["diesel_kw"]
        + output["discharge_kw"] + output["unserved_kw"] + output["pv_used_kw"]
        - output["charge_kw"] - output["demand_kw"]
    )

    energy_demand = float(output["demand_kw"].sum() * dt)
    unmet = float(output["unserved_kw"].sum() * dt)
    grid_energy = float(output["grid_kw"].sum() * dt)
    gas_energy = float(output["gas_kw"].sum() * dt)
    diesel_energy = float(output["diesel_kw"].sum() * dt)
    wear_energy = float((output["charge_kw"] + output["discharge_kw"]).sum() * dt)
    energy_cost = (
        grid_energy * scenario.grid_cost_eur_kwh
        + gas_energy * scenario.gas_cost_eur_kwh
        + diesel_energy * scenario.diesel_cost_eur_kwh
        + wear_energy * scenario.battery_wear_eur_kwh
    )
    emissions = (
        grid_energy * scenario.grid_emission_kg_kwh
        + gas_energy * scenario.gas_emission_kg_kwh
        + diesel_energy * scenario.diesel_emission_kg_kwh
    )
    metrics = {
        "demand_kwh": energy_demand,
        "unserved_kwh": unmet,
        "served_kwh": energy_demand - unmet,
        "served_pct": 100 * (1 - unmet / energy_demand) if energy_demand > 0 else 100.0,
        "energy_cost_eur": float(energy_cost),
        "emissions_kg": float(emissions),
        "grid_kwh": grid_energy,
        "gas_kwh": gas_energy,
        "diesel_kwh": diesel_energy,
        "pv_used_kwh": float(output["pv_used_kw"].sum() * dt),
        "curtailed_pv_kwh": float(output["curtailed_pv_kw"].sum() * dt),
        "bess_throughput_kwh": wear_energy,
        "terminal_soc_kwh": float(output["soc_kwh"].iloc[-1]),
        "objective_eur": float(solution.fun),
    }
    return DispatchResult(output, metrics)
