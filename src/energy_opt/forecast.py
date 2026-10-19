"""Demanda sintética y pronóstico de un día mediante Machine Learning.

IMPORTANTE: los datos son simulados. No representan mediciones de clientes.
"""

from datetime import date, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

FEATURES = ["hour_sin", "hour_cos", "weekday", "temperature_c", "weekend"]


def demo_temperature(index: pd.DatetimeIndex) -> np.ndarray:
    """Temperatura sintética conocida; equivale a un pronóstico meteorológico idealizado."""
    hour = index.hour.to_numpy()
    day = index.dayofyear.to_numpy()
    return 22 + 6 * np.sin(2 * np.pi * (hour - 9) / 24) + 3 * np.sin(2 * np.pi * day / 365.25)


def calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Solo variables conocidas antes del intervalo que se pronostica."""
    dates = pd.to_datetime(frame["timestamp"])
    hours = dates.dt.hour
    weekdays = dates.dt.dayofweek
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * hours / 24).to_numpy(),
        "hour_cos": np.cos(2 * np.pi * hours / 24).to_numpy(),
        "weekday": weekdays.to_numpy(),
        "temperature_c": frame["temperature_c"].to_numpy(),
        "weekend": (weekdays >= 5).astype(int).to_numpy(),
    }, index=frame.index)


def make_demo_history(target_day: date, days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Crea una serie horaria reproducible, estrictamente anterior al día objetivo."""
    if days < 14:
        raise ValueError("Se necesitan al menos 14 días de entrenamiento")
    start = pd.Timestamp(target_day - timedelta(days=days))
    timestamps = pd.date_range(start, periods=days * 24, freq="h")
    rng = np.random.default_rng(seed)
    temp = demo_temperature(timestamps) + rng.normal(0, 1.0, len(timestamps))
    hours = timestamps.hour.to_numpy()
    workday = (timestamps.dayofweek.to_numpy() < 5).astype(float)
    work_hours = ((hours >= 8) & (hours < 18)).astype(float)
    evening = ((hours >= 18) & (hours < 23)).astype(float)
    # Curva con rasgos industriales, sin pretensión de reproducir un activo real.
    demand = (45 + 18 * workday * work_hours + 19 * evening
              + 1.0 * (temp - 20) + rng.normal(0, 3.0, len(timestamps)))
    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature_c": temp,
        "demand_kw": np.maximum(10.0, demand),
    })


def forecast_day(target_day: date, history_days: int = 120) -> tuple[pd.DataFrame, dict[str, float]]:
    """Valida temporalmente y entrena un modelo final para pronosticar 24 h.

    El modelo de evaluación ve exclusivamente el primer 80 % de la historia.
    El modelo final se reentrena con toda la historia disponible.
    """
    history = make_demo_history(target_day, days=history_days)
    x = calendar_features(history)
    y = history["demand_kw"].to_numpy()
    split = int(len(history) * 0.8)

    def make_model() -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_iter=110, learning_rate=0.09, max_leaf_nodes=15,
            min_samples_leaf=18, random_state=42,
        )

    validation_model = make_model()
    validation_model.fit(x.iloc[:split][FEATURES], y[:split])
    predictions = validation_model.predict(x.iloc[split:][FEATURES])
    mae = float(mean_absolute_error(y[split:], predictions))
    # Comparador de persistencia estacional: misma hora del día anterior.
    naive = y[np.arange(split, len(y)) - 24]
    naive_mae = float(mean_absolute_error(y[split:], naive))

    final_model = make_model()
    final_model.fit(x[FEATURES], y)
    future_times = pd.date_range(pd.Timestamp(target_day), periods=24, freq="h")
    future = pd.DataFrame({
        "timestamp": future_times,
        "temperature_c": demo_temperature(future_times),
    })
    future["demand_kw"] = np.maximum(0, final_model.predict(calendar_features(future)[FEATURES]))
    metrics = {"validation_mae_kw": mae, "naive_mae_kw": naive_mae,
               "train_hours": float(split), "test_hours": float(len(y) - split)}
    return future, metrics


def add_solar_and_outages(forecast: pd.DataFrame, pv_peak_kw: float,
                          outage_hours: tuple[int, ...]) -> pd.DataFrame:
    """Añade irradiancia idealizada y disponibilidad de red al pronóstico."""
    result = forecast.copy()
    hour = pd.to_datetime(result["timestamp"]).dt.hour.to_numpy(dtype=float)
    # Perfil sinusoidal 06:00–18:00; factor de pérdidas demostrativo 0.80.
    solar_shape = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12))
    result["pv_kw"] = pv_peak_kw * 0.80 * solar_shape
    result["grid_available"] = ~pd.to_datetime(result["timestamp"]).dt.hour.isin(outage_hours)
    return result
