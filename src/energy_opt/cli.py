"""CLI reproducible para validación y automatización local."""

import argparse
from datetime import date, timedelta
from pathlib import Path

from energy_opt.pipeline import run_demo
from energy_opt.scenario import Scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Despacho demo PV+BESS+gas+diésel con ML")
    parser.add_argument("--date", type=date.fromisoformat,
                        default=date.today() + timedelta(days=1), help="Fecha ISO (YYYY-MM-DD)")
    parser.add_argument("--outages", type=int, nargs="*", default=[18, 19, 20],
                        help="Horas sin red de 0 a 23")
    parser.add_argument("--output", type=Path, default=None,
                        help="Ruta CSV opcional para resultados horarios")
    args = parser.parse_args()
    scenario = Scenario(outage_hours=tuple(args.outages))
    solved, reference, metrics = run_demo(args.date, scenario)
    print(f"Fecha del escenario: {args.date} (datos sintéticos)")
    print(f"MAE ML (validación temporal): {metrics['validation_mae_kw']:.2f} kW")
    print(f"MAE persistencia 24 h: {metrics['naive_mae_kw']:.2f} kW")
    print(f"Demanda: {solved.metrics['demand_kwh']:.2f} kWh")
    print(f"Carga servida con BESS: {solved.metrics['served_pct']:.2f}%")
    print(f"Energía no servida con BESS: {solved.metrics['unserved_kwh']:.2f} kWh")
    print(f"Energía no servida sin BESS: {reference.metrics['unserved_kwh']:.2f} kWh")
    print(f"Coste operativo con BESS: {solved.metrics['energy_cost_eur']:.2f} EUR")
    print(f"Coste operativo sin BESS: {reference.metrics['energy_cost_eur']:.2f} EUR")
    print("AVISO: simulación sintética, no apta para control de equipos.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        solved.timeseries.to_csv(args.output, index=False)
        print(f"CSV exportado: {args.output}")


if __name__ == "__main__":
    main()
