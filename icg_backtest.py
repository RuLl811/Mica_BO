"""
=============================================================================
 ICG — Pipeline de evaluacion predictiva univariada a horizonte fijo t+3
=============================================================================
Backtesting con recursion de origenes (rolling-origin / time series CV).

Diseno:
  * Horizonte objetivo FIJO: se evalua unicamente el paso h=3. Los pasos
    h=1 y h=2 se generan pero se descartan, porque promediar horizontes
    contamina la metrica del objetivo.
  * Sin fuga de informacion: en cada origen T el modelo se re-estima
    integramente con y[:T] y se proyecta a T+3. No hay parametros
    estimados sobre datos futuros.
  * Dos esquemas de muestra: expanding (ventana creciente) y rolling
    (ventana fija), para detectar dependencia de la conclusion respecto
    del tratamiento del pasado lejano.
  * Comparacion contra benchmarks obligatorios (random walk, media,
    drift) y test de Diebold-Mariano con correccion de muestra chica.

Uso:
    python icg_backtest.py --datos ICG.xlsx --h 3 --min-train 120

Autor: pipeline cuantitativo — statsmodels 0.15 / pandas 2.x+
=============================================================================
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# 1. Carga y validacion de datos
# --------------------------------------------------------------------------
def cargar_icg(path: str, col_fecha: str = "Fecha", col_valor: str = "ICG") -> pd.Series:
    """Lee el ICG y devuelve una Serie mensual regular a fin de mes.

    Valida frecuencia, ordena, y falla ruidosamente si hay huecos: un
    backtest sobre un indice irregular produce horizontes h que no son
    comparables entre origenes.
    """
    df = pd.read_excel(path)
    s = df.set_index(col_fecha)[col_valor].astype(float).sort_index()
    s.index = pd.PeriodIndex(pd.to_datetime(s.index), freq="M").to_timestamp("M")
    s = s.asfreq("ME")
    s.name = "ICG"

    if s.isna().any():
        huecos = s[s.isna()].index.strftime("%Y-%m").tolist()
        raise ValueError(f"Huecos en la serie mensual: {huecos}")
    if s.index.duplicated().any():
        raise ValueError("Fechas duplicadas en el indice")
    return s


# --------------------------------------------------------------------------
# 2. Catalogo de predictores
#    Contrato: fn(y_train: pd.Series, h: int) -> float  (prediccion en t+h)
# --------------------------------------------------------------------------
def _sarimax_fc(y, h, order, trend):
    r = SARIMAX(y, order=order, trend=trend,
                enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
    return float(r.forecast(h).iloc[-1])


def f_naive(y, h):
    """Random walk: E[y_{T+h}] = y_T. Benchmark obligatorio."""
    return float(y.iloc[-1])


def f_media(y, h):
    """Reversion total a la media muestral. Cota superior de reversion."""
    return float(y.mean())


def f_drift(y, h):
    """Random walk con deriva estimada por el promedio de las variaciones."""
    return float(y.iloc[-1] + h * (y.iloc[-1] - y.iloc[0]) / (len(y) - 1))


def f_ar1(y, h):
    """ARIMA(1,0,0) con constante — reversion parcial a la media incondicional."""
    return _sarimax_fc(y, h, (1, 0, 0), "c")


def f_ar2(y, h):
    return _sarimax_fc(y, h, (2, 0, 0), "c")


def f_arma11(y, h):
    return _sarimax_fc(y, h, (1, 0, 1), "c")


def f_ima11(y, h):
    """ARIMA(0,1,1) — equivalente en forma reducida al suavizado exponencial simple."""
    return _sarimax_fc(y, h, (0, 1, 1), "n")


def f_arima111(y, h):
    return _sarimax_fc(y, h, (1, 1, 1), "n")


def f_arima_auto(y, h, p_max=2, q_max=2):
    """Seleccion por AICc dentro de cada ventana (p<=2, d in {0,1}, q<=2).

    Se incluye deliberadamente para cuantificar el costo predictivo de la
    inestabilidad de seleccion: re-elegir el orden en cada origen agrega
    varianza que no compensa el sesgo que evita.
    """
    mejor, mejor_ic = None, np.inf
    for d in (0, 1):
        for p in range(p_max + 1):
            for q in range(q_max + 1):
                if (p, d, q) == (0, 0, 0):
                    continue
                try:
                    r = SARIMAX(y, order=(p, d, q), trend="c" if d == 0 else "n",
                                enforce_stationarity=True,
                                enforce_invertibility=True).fit(disp=False)
                    k, n = r.params.size, r.nobs
                    ic = r.aic + 2 * k * (k + 1) / max(n - k - 1, 1)
                    if ic < mejor_ic:
                        mejor_ic, mejor = ic, r
                except Exception:
                    continue
    return float(mejor.forecast(h).iloc[-1])


def f_ses(y, h):
    """Suavizado exponencial simple (ETS A,N,N)."""
    r = ETSModel(y, error="add", trend=None, seasonal=None).fit(disp=False)
    return float(r.forecast(h).iloc[-1])


def f_holt_amort(y, h):
    """Holt con tendencia amortiguada (ETS A,Ad,N)."""
    r = ETSModel(y, error="add", trend="add", damped_trend=True,
                 seasonal=None).fit(disp=False)
    return float(r.forecast(h).iloc[-1])


def f_holt(y, h):
    r = ETSModel(y, error="add", trend="add", damped_trend=False,
                 seasonal=None).fit(disp=False)
    return float(r.forecast(h).iloc[-1])


def f_uc_nivel(y, h):
    """Componentes no observables: nivel local (filtro de Kalman)."""
    r = UnobservedComponents(y, level="local level").fit(disp=False)
    return float(r.forecast(h).iloc[-1])


def f_uc_tendencia(y, h):
    r = UnobservedComponents(y, level="local linear trend").fit(disp=False)
    return float(r.forecast(h).iloc[-1])


def f_uc_nivel_ar1(y, h):
    """Nivel local estocastico + componente AR(1) irregular.

    Descompone el ICG en un nivel permanente y un desvio transitorio
    mean-reverting: la lectura estructural natural de un indice de
    confianza (regimen politico vs. ruido coyuntural).
    """
    r = UnobservedComponents(y, level="local level", autoregressive=1).fit(disp=False)
    return float(r.forecast(h).iloc[-1])


def f_ar_directo(y, h, p=3):
    """Proyeccion DIRECTA: OLS de y_{t+h} sobre {y_t, ..., y_{t-p+1}}.

    Optimiza la perdida al horizonte objetivo en lugar de iterar el paso
    a un paso; es robusto a mala especificacion dinamica, a costa de
    menos eficiencia si el modelo iterado esta bien especificado.
    """
    Y = y.to_numpy()
    n = len(Y)
    X = np.array([Y[t - p + 1:t + 1][::-1] for t in range(p - 1, n - h)])
    tgt = np.array([Y[t + h] for t in range(p - 1, n - h)])
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, tgt, rcond=None)
    return float(np.r_[1.0, Y[n - p:][::-1]] @ beta)


def f_theta(y, h):
    """Metodo Theta (Assimakopoulos-Nikolopoulos): SES + media deriva lineal."""
    r = ETSModel(y, error="add", trend=None, seasonal=None).fit(disp=False)
    alpha = float(dict(zip(r.model.param_names, np.asarray(r.params)))["smoothing_level"])
    n = len(y)
    b = np.polyfit(np.arange(n), y.to_numpy(), 1)[0]
    return float(r.forecast(h).iloc[-1]) + 0.5 * b * (
        h - 1 + (1 - (1 - alpha) ** n) / max(alpha, 1e-6))


CATALOGO: dict[str, Callable] = {
    # -- benchmarks
    "Naive (RW)": f_naive,
    "Media historica": f_media,
    "Drift": f_drift,
    "Theta": f_theta,
    # -- suavizado exponencial
    "SES (ETS A,N,N)": f_ses,
    "Holt (ETS A,A,N)": f_holt,
    "Holt amortiguado (A,Ad,N)": f_holt_amort,
    # -- ARIMA
    "ARIMA(1,0,0)+c": f_ar1,
    "ARIMA(2,0,0)+c": f_ar2,
    "ARIMA(1,0,1)+c": f_arma11,
    "ARIMA(0,1,1)": f_ima11,
    "ARIMA(1,1,1)": f_arima111,
    "ARIMA auto (AICc)": f_arima_auto,
    # -- espacio de estados / Kalman
    "UC nivel local": f_uc_nivel,
    "UC tendencia local": f_uc_tendencia,
    "UC nivel + AR(1)": f_uc_nivel_ar1,
    # -- proyeccion directa
    "AR directo h (p=3)": f_ar_directo,
}


# --------------------------------------------------------------------------
# 3. Motor de backtesting
# --------------------------------------------------------------------------
@dataclass
class ResultadoBacktest:
    predicciones: pd.DataFrame          # columnas = modelos, indice = fecha objetivo
    realizado: pd.Series
    metricas: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.metricas = self._metricas()

    def _metricas(self) -> pd.DataFrame:
        y = self.realizado.to_numpy()
        filas = []
        for m in self.predicciones.columns:
            e = y - self.predicciones[m].to_numpy()
            filas.append({
                "modelo": m,
                "RMSE": np.sqrt(np.nanmean(e ** 2)),
                "MAE": np.nanmean(np.abs(e)),
                "MAPE_%": 100 * np.nanmean(np.abs(e / y)),
                "sesgo": np.nanmean(e),
                "n": int(np.sum(~np.isnan(e))),
            })
        out = pd.DataFrame(filas).set_index("modelo").sort_values("RMSE")
        if "Naive (RW)" in out.index:
            out["U_Theil"] = out["RMSE"] / out.loc["Naive (RW)", "RMSE"]
            out["DM_p_vs_naive"] = [
                diebold_mariano(
                    y - self.predicciones["Naive (RW)"].to_numpy(),
                    y - self.predicciones[m].to_numpy())[1]
                for m in out.index]
        return out


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 3, potencia: int = 2):
    """DM sobre la perdida |e|^potencia con correccion Harvey-Leybourne-Newbold.

    H0: igual precision predictiva. Estadistico > 0 favorece al modelo 2.
    La varianza de larga duracion usa h-1 rezagos porque las predicciones
    solapadas a h pasos generan errores MA(h-1) por construccion.
    """
    d = np.abs(e1) ** potencia - np.abs(e2) ** potencia
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:n - k] - dbar)) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        return np.nan, np.nan
    correccion = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = correccion * dbar / np.sqrt(var)
    return stat, 2 * (1 - stats.t.cdf(abs(stat), n - 1))


def backtest(serie: pd.Series, modelos: dict[str, Callable], h: int = 3,
             min_train: int = 120, esquema: str = "expanding",
             ventana: int | None = None, verbose: bool = True) -> ResultadoBacktest:
    """Recursion de origenes evaluando exclusivamente el paso t+h."""
    if esquema not in ("expanding", "rolling"):
        raise ValueError("esquema debe ser 'expanding' o 'rolling'")
    ventana = ventana or min_train
    origenes = range(min_train, len(serie) - h + 1)
    fechas_obj = [serie.index[t + h - 1] for t in origenes]

    preds = {}
    for nombre, fn in modelos.items():
        col = []
        for t in origenes:
            y_tr = serie.iloc[:t] if esquema == "expanding" else serie.iloc[t - ventana:t]
            try:
                col.append(fn(y_tr, h))
            except Exception:
                col.append(np.nan)          # el origen fallido no contamina el resto
        preds[nombre] = col
        if verbose:
            fallos = int(np.sum(np.isnan(col)))
            print(f"  {nombre:30s} listo ({len(col)} origenes, {fallos} fallos)")

    return ResultadoBacktest(
        predicciones=pd.DataFrame(preds, index=pd.Index(fechas_obj, name="fecha_objetivo")),
        realizado=serie.reindex(fechas_obj))


def estabilidad_por_subperiodo(rb: ResultadoBacktest, cortes: list[tuple[str, str]],
                               modelos: list[str] | None = None) -> pd.DataFrame:
    """RMSE del paso t+h por tramo: detecta si el ranking depende del regimen."""
    modelos = modelos or list(rb.predicciones.columns)
    out = {}
    for m in modelos:
        e = rb.realizado - rb.predicciones[m]
        out[m] = {f"{a[:7]}/{b[:7]}": np.sqrt(np.nanmean(e[a:b].to_numpy() ** 2))
                  for a, b in cortes}
    return pd.DataFrame(out).T


# --------------------------------------------------------------------------
# 4. Ejecucion
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Backtest t+h del ICG")
    ap.add_argument("--datos", default="ICG.xlsx")
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--min-train", type=int, default=120)
    ap.add_argument("--rapido", action="store_true",
                    help="excluye 'ARIMA auto (AICc)' (costoso)")
    ap.add_argument("--salida", default="backtest_icg.xlsx")
    args = ap.parse_args()

    s = cargar_icg(args.datos)
    print(f"Serie ICG: {len(s)} obs | {s.index.min():%Y-%m} a {s.index.max():%Y-%m}\n")

    modelos = dict(CATALOGO)
    if args.rapido:
        modelos.pop("ARIMA auto (AICc)", None)

    resultados = {}
    for esquema in ("expanding", "rolling"):
        print(f"[{esquema}] evaluando h={args.h} ...")
        rb = backtest(s, modelos, h=args.h, min_train=args.min_train, esquema=esquema)
        resultados[esquema] = rb
        print(f"\n--- Metricas h={args.h} | esquema {esquema} | "
              f"{len(rb.realizado)} origenes ---")
        print(rb.metricas.round(4).to_string(), "\n")

    rb = resultados["expanding"]
    print("--- Estabilidad del RMSE por subperiodo (expanding) ---")
    print(estabilidad_por_subperiodo(
        rb, [("2014-01", "2017-12"), ("2018-01", "2021-12"), ("2022-01", "2026-12")],
        modelos=rb.metricas.index[:6].tolist() + ["Naive (RW)"]).round(4).to_string())

    with pd.ExcelWriter(args.salida) as xls:
        for esquema, r in resultados.items():
            r.metricas.round(6).to_excel(xls, sheet_name=f"metricas_{esquema}")
            r.predicciones.round(6).to_excel(xls, sheet_name=f"preds_{esquema}")
        rb.realizado.round(6).to_frame("ICG_real").to_excel(xls, sheet_name="realizado")
    print(f"\nResultados exportados a {args.salida}")


if __name__ == "__main__":
    main()
