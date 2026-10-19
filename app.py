"""Panel Streamlit de simulación; no requiere API keys ni servicios cloud."""

from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from energy_opt.dispatch import optimize_dispatch
from energy_opt.forecast import add_solar_and_outages, forecast_day
from energy_opt.scenario import Scenario
from dataclasses import replace

st.set_page_config(page_title="Industrial Energy AI | PV + BESS + Backup",
                   page_icon="⚡", layout="wide")


@st.cache_data(show_spinner="Entrenando el modelo demostrativo...")
def cached_forecast(target_date: date):
    # Solo se cachea el pronóstico; los escenarios se optimizan en cada cambio.
    return forecast_day(target_date)


st.title("Industrial Energy AI")
st.caption("PV · Battery storage · Gas generation · Diesel backup · Day-ahead ML forecasting")
st.warning("DEMO CON DATOS SINTÉTICOS. No usar para maniobras, control de equipos ni decisiones operativas reales.")

with st.sidebar:
    st.header("Escenario")
    target_date = st.date_input("Día a simular", value=date.today() + timedelta(days=1))
    pv_kw = st.slider("Solar fotovoltaica (kWp)", 0, 200, 60, 5)
    bess_kwh = st.slider("Capacidad BESS (kWh)", 0, 400, 90, 10)
    bess_kw = st.slider("Potencia BESS (kW)", 0, 160, 40, 5)
    gas_kw = st.slider("Respaldo a gas (kW)", 0, 200, 50, 5)
    diesel_kw = st.slider("Respaldo diésel (kW)", 0, 200, 75, 5)
    outage = st.multiselect("Horas sin red", options=list(range(24)), default=[18, 19, 20])
    carbon_price = st.slider("Precio carbono (EUR/tCO₂)", 0, 250, 0, 10)
    with st.expander("Costes de energía (supuestos)"):
        grid_cost = st.number_input("Red (EUR/kWh)", min_value=0.0, value=0.22, step=0.01)
        gas_cost = st.number_input("Gas (EUR/kWh)", min_value=0.0, value=0.19, step=0.01)
        diesel_cost = st.number_input("Diésel (EUR/kWh)", min_value=0.0, value=0.34, step=0.01)
    st.caption("Los precios, emisiones y capacidades son parámetros ilustrativos, no fichas técnicas Jenbacher ni ofertas comerciales.")

scenario = Scenario(
    pv_peak_kw=pv_kw, battery_capacity_kwh=bess_kwh, battery_power_kw=bess_kw,
    gas_max_kw=gas_kw, diesel_max_kw=diesel_kw,
    outage_hours=tuple(sorted(outage)), carbon_price_eur_tonne=carbon_price,
    grid_cost_eur_kwh=grid_cost, gas_cost_eur_kwh=gas_cost,
    diesel_cost_eur_kwh=diesel_cost,
)

try:
    demand, quality = cached_forecast(target_date)
    profile = add_solar_and_outages(demand, scenario.pv_peak_kw, scenario.outage_hours)
    solved = optimize_dispatch(profile, scenario)
    baseline = optimize_dispatch(profile, replace(scenario, battery_capacity_kwh=0,
                                                   battery_power_kw=0))
except (ValueError, RuntimeError) as exc:
    st.error(f"No pudo simularse el escenario: {exc}")
    st.stop()

m = solved.metrics
b = baseline.metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Carga servida", f"{m['served_pct']:.2f}%")
c2.metric("Energía no servida", f"{m['unserved_kwh']:.1f} kWh",
          delta=f"{b['unserved_kwh'] - m['unserved_kwh']:+.1f} kWh evitados")
c3.metric("Coste operativo", f"€{m['energy_cost_eur']:.2f}")
c4.metric("Emisiones estimadas", f"{m['emissions_kg']:.1f} kg CO₂")

st.subheader("Despacho horario de potencia")
trace = solved.timeseries
plot = go.Figure()
for name, label, color in (
    ("pv_used_kw", "FV utilizada", "#eab308"),
    ("grid_kw", "Red", "#4f82c2"),
    ("gas_kw", "Generador gas", "#10b981"),
    ("diesel_kw", "Generador diésel", "#f97316"),
    ("discharge_kw", "Descarga BESS", "#8b5cf6"),
    ("unserved_kw", "Carga no servida", "#ef4444"),
):
    plot.add_trace(go.Scatter(x=trace["timestamp"], y=trace[name],
                              mode="lines", stackgroup="supply", name=label,
                              line={"width": 0.5, "color": color}))
plot.add_trace(go.Scatter(x=trace["timestamp"], y=trace["demand_kw"],
                          name="Demanda pronosticada", mode="lines",
                          line={"color": "#f8fafc", "width": 2, "dash": "dash"}))
plot.update_layout(template="plotly_dark", xaxis_title="Hora", yaxis_title="Potencia (kW)",
                   legend={"orientation": "h", "y": -0.25}, margin={"l": 10, "r": 10, "t": 5})
st.plotly_chart(plot, use_container_width=True)
st.caption("El gráfico apila fuentes de suministro y carga no servida. La carga BESS está excluida de las fuentes porque consume energía.")

left, right = st.columns(2)
with left:
    st.subheader("Estado de carga del BESS")
    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(x=trace["timestamp"], y=trace["soc_pct"],
                                 mode="lines+markers", name="SOC (%)",
                                 line={"color": "#14b8a6", "width": 3}))
    fig_soc.update_layout(template="plotly_dark", yaxis_title="SOC (%)", yaxis_range=[0, 100],
                          margin={"l": 10, "r": 10, "t": 5})
    st.plotly_chart(fig_soc, use_container_width=True)
with right:
    st.subheader("Pronóstico ML · validación temporal")
    st.metric("Error absoluto medio del modelo", f"{quality['validation_mae_kw']:.2f} kW")
    st.metric("Error baseline 24 h", f"{quality['naive_mae_kw']:.2f} kW")
    st.caption("Modelo HistGradientBoostingRegressor entrenado con datos sintéticos de 120 días. Validación: último 20 % de horas, sin mezcla temporal.")

st.subheader("Comparativa de escenarios (simulada)")
st.dataframe({
    "Indicador": ["Carga no servida (kWh)", "Coste operativo (EUR)", "Emisiones (kg CO₂)"],
    "Con BESS": [round(m["unserved_kwh"], 2), round(m["energy_cost_eur"], 2),
                 round(m["emissions_kg"], 2)],
    "Sin BESS": [round(b["unserved_kwh"], 2), round(b["energy_cost_eur"], 2),
                 round(b["emissions_kg"], 2)],
}, hide_index=True, use_container_width=True)
st.caption("Comparación hipotética de despacho, no estimación de ahorro real. El coste operativo excluye la penalización económica de energía no servida y CAPEX/OPEX fijo.")

st.subheader("Resultados horarios")
st.dataframe(trace.round(3), use_container_width=True, hide_index=True)
st.download_button("Descargar despacho CSV", data=trace.to_csv(index=False),
                   file_name=f"energy-dispatch-{target_date}.csv", mime="text/csv")

with st.expander("Supuestos y limitaciones del modelo"):
    st.markdown("""
- Red disponible excepto en las horas seleccionadas; sin exportación a red.
- Paso temporal de una hora; energía calculada como potencia × 1 h.
- Restricciones de potencia BESS, SOC mínimo/máximo y eficiencias de carga/descarga.
- El SOC terminal es al menos igual al inicial, evitando energía inicial gratuita.
- Optimización lineal continua (HiGHS); **no** implementa compromiso de unidades ni rampas.
- Generadores a gas y diésel genéricos; **no** contiene curvas Jenbacher certificadas.
- No se modelan start-stop, mínimo técnico, fallas, combustible, degradación no lineal ni reserva girante.
- La relajación lineal no impide matemáticamente carga y descarga simultáneas en todos los precios.
- PV, demanda, temperatura, costes y factores de emisiones son **hipótesis sintéticas**.
- El modelo ML tiene prueba temporal y baseline, pero **no** está validado contra datos industriales.
- Este prototipo no es un controlador SCADA/EMS ni sustituye ingeniería de detalle o protecciones.
""")
