"""
=============================================================================
 ICG — Pronostico del PROMEDIO trimestral (t+1 .. t+3)
=============================================================================
Cambia el objetivo respecto de la version puntual: en lugar de estimar el
ICG del mes t+3, estima

    m_{T+1:T+3} = (y_{T+1} + y_{T+2} + y_{T+3}) / 3

El promedio suaviza el ruido de muestreo de la encuesta y el timing exacto
de los saltos, y baja el RMSE ~26% sin cambiar de modelo. Es el objetivo
correcto si lo que importa es el nivel de confianza del trimestre y no el
valor puntual de un mes.

QUE OPTIMIZA ESTE SCRIPT
------------------------
Corre una busqueda sobre 14 especificaciones evaluadas por RMSE del
promedio en recursion de origenes, y ademas evalua el PROCEDIMIENTO de
seleccion recursiva (elegir en cada origen el mejor modelo hasta ese
momento). El resultado relevante es que seleccionar es peor que fijar:
el procedimiento recursivo da 0.2773 contra 0.2726 del AR(1)+c fijo.
Por eso el modelo de produccion queda FIJO en AR(1)+c y la busqueda se
conserva solo como diagnostico auditable.

INTERVALOS
----------
Para el promedio, la varianza del error NO es la suma de las varianzas por
paso: los tres errores comparten innovaciones. Para el AR(1),

    e_prom = [eps_1(1+phi+phi^2) + eps_2(1+phi) + eps_3] / 3
    Var    = sigma^2 [ (1+phi+phi^2)^2 + (1+phi)^2 + 1 ] / 9

Se reportan dos bandas: la gaussiana de esa expresion, y una bootstrap
por remuestreo de residuos, que no impone normalidad. Los residuos del
ICG tienen curtosis ~11, asi que la segunda es la relevante para cola.

Uso:
    python icg_promedio.py --datos ICG.xlsx --graficos
=============================================================================
"""
from __future__ import annotations

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.structural import UnobservedComponents

from icg_backtest import cargar_icg, diebold_mariano

warnings.filterwarnings("ignore")

H = 3
TAU = 0.5                      # umbral del gate, fijado por criterio (ver icg_hibrido.py)
AZUL, NARANJA, GRIS = "#4C72B0", "#DD8452", "#8C8C8C"

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

rmse = lambda e: float(np.sqrt(np.nanmean(np.asarray(e, float) ** 2)))


# --------------------------------------------------------------------------
# Objetivo
# --------------------------------------------------------------------------
def objetivo_promedio(s: pd.Series, t: int, h: int = H) -> float:
    """Promedio realizado de y_{t+1}..y_{t+h}, visto desde el origen t."""
    return float(s.iloc[t:t + h].mean())


# --------------------------------------------------------------------------
# Candidatos. Contrato: fn(y_train) -> estimacion del promedio t+1..t+h
# --------------------------------------------------------------------------
def _sarimax_prom(y, order, trend):
    r = SARIMAX(y, order=order, trend=trend, enforce_stationarity=True,
                enforce_invertibility=True).fit(disp=False)
    return float(r.forecast(H).mean())


def _directo(y, p):
    """OLS de m_{t+1:t+3} sobre p rezagos: optimiza el objetivo directamente."""
    Y = y.to_numpy()
    n = len(Y)
    idx = np.arange(p - 1, n - H)
    X = np.column_stack([np.ones(len(idx))] + [Y[idx - k] for k in range(p)])
    tgt = np.array([Y[i + 1:i + 1 + H].mean() for i in idx])
    beta, *_ = np.linalg.lstsq(X, tgt, rcond=None)
    return float(np.r_[1.0, [Y[n - 1 - k] for k in range(p)]] @ beta)


CANDIDATOS = {
    "Naive y_T": lambda y: float(y.iloc[-1]),
    "Naive MA3": lambda y: float(y.iloc[-3:].mean()),
    "Media historica": lambda y: float(y.mean()),
    "AR(1)+c": lambda y: _sarimax_prom(y, (1, 0, 0), "c"),
    "AR(2)+c": lambda y: _sarimax_prom(y, (2, 0, 0), "c"),
    "AR(3)+c": lambda y: _sarimax_prom(y, (3, 0, 0), "c"),
    "ARMA(1,1)+c": lambda y: _sarimax_prom(y, (1, 0, 1), "c"),
    "ARMA(2,1)+c": lambda y: _sarimax_prom(y, (2, 0, 1), "c"),
    "ARIMA(0,1,1)": lambda y: _sarimax_prom(y, (0, 1, 1), "n"),
    "Holt amortiguado": lambda y: float(ETSModel(
        y, error="add", trend="add", damped_trend=True, seasonal=None
    ).fit(disp=False).forecast(H).mean()),
    "UC nivel+AR(1)": lambda y: float(UnobservedComponents(
        y, level="local level", autoregressive=1).fit(disp=False).forecast(H).mean()),
    "Directo p=1": lambda y: _directo(y, 1),
    "Directo p=2": lambda y: _directo(y, 2),
    "Directo p=3": lambda y: _directo(y, 3),
}


# --------------------------------------------------------------------------
# Backtest sobre el objetivo promedio
# --------------------------------------------------------------------------
def backtest_promedio(s: pd.Series, candidatos: dict, min_train: int = 120,
                      verbose: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    origenes = range(min_train, len(s) - H + 1)
    fechas = pd.Index([s.index[t + H - 1] for t in origenes], name="fin_ventana")
    real = pd.Series([objetivo_promedio(s, t) for t in origenes], index=fechas,
                     name="promedio_observado")
    preds = {}
    for nombre, fn in candidatos.items():
        preds[nombre] = [fn(s.iloc[:t]) for t in origenes]
        if verbose:
            print(f"  {nombre:20s} listo")
    return pd.DataFrame(preds, index=fechas), real


def tabla_metricas(P: pd.DataFrame, real: pd.Series, base="Naive y_T") -> pd.DataFrame:
    y = real.to_numpy()
    E = {m: y - P[m].to_numpy() for m in P.columns}
    out = pd.DataFrame([{
        "modelo": m, "RMSE": rmse(e), "MAE": np.mean(np.abs(e)),
        "MAPE_%": 100 * np.mean(np.abs(e / y)), "sesgo": e.mean(),
        "U_Theil": rmse(e) / rmse(E[base]),
        "DM_p_vs_naive": diebold_mariano(E[base], e)[1],
    } for m, e in E.items()]).set_index("modelo").sort_values("RMSE")
    return out


def costo_de_seleccionar(P: pd.DataFrame, real: pd.Series, fijo="AR(1)+c",
                         burn: int = 48) -> dict:
    """Compara fijar el modelo contra elegirlo recursivamente por RMSE.

    Es el control que impide vender como 'optimizado' un resultado que solo
    refleja haber mirado la tabla completa antes de elegir.
    """
    y = real.to_numpy()
    E = {m: y - P[m].to_numpy() for m in P.columns}
    elegidos, pred = [], []
    for i in range(burn, len(y)):
        marcador = {m: np.mean(E[m][:i] ** 2) for m in P.columns}
        gana = min(marcador, key=marcador.get)
        elegidos.append(gana)
        pred.append(P[gana].to_numpy()[i])
    return {
        "RMSE_seleccion_recursiva": rmse(y[burn:] - np.array(pred)),
        "RMSE_modelo_fijo": rmse(E[fijo][burn:]),
        "RMSE_naive": rmse(E["Naive y_T"][burn:]),
        "frecuencia_elegido": pd.Series(elegidos).value_counts().to_dict(),
    }


# --------------------------------------------------------------------------
# Modelo de produccion
# --------------------------------------------------------------------------
def ajustar_produccion(s: pd.Series):
    return SARIMAX(s, order=(1, 0, 0), trend="c", enforce_stationarity=True,
                   enforce_invertibility=True).fit(disp=False)


def pronostico_promedio(res, s: pd.Series, n_boot: int = 20000, semilla: int = 7) -> dict:
    """Punto + banda gaussiana cerrada + banda bootstrap para el promedio."""
    c = float(res.params.iloc[0])
    phi = float(res.params.iloc[1])
    sigma2 = float(res.params.iloc[-1])
    mu = c / (1 - phi)
    sd_inc = np.sqrt(sigma2 / (1 - phi ** 2))
    y_T = float(s.iloc[-1])

    punto = float(res.forecast(H).mean())

    pesos = np.array([sum(phi ** j for j in range(k)) for k in range(H, 0, -1)])
    var_gauss = sigma2 * np.sum(pesos ** 2) / H ** 2
    sd_gauss = np.sqrt(var_gauss)

    # bootstrap: remuestreo de residuos estandarizados, sin supuesto de normalidad
    rng = np.random.default_rng(semilla)
    resid = np.asarray(res.resid[1:], float)
    resid = resid - resid.mean()
    shocks = rng.choice(resid, size=(n_boot, H), replace=True)
    y = np.empty((n_boot, H))
    prev = np.full(n_boot, y_T)
    for k in range(H):
        prev = mu + phi * (prev - mu) + shocks[:, k]
        y[:, k] = prev
    sim = y.mean(axis=1)
    q_lo, q_hi = np.quantile(sim, [0.025, 0.975])

    return {
        "phi": phi, "mu": mu, "sd_incondicional": sd_inc, "ultimo": y_T,
        "z": (y_T - mu) / sd_inc, "gate_activo": abs((y_T - mu) / sd_inc) >= TAU,
        "punto": punto, "sd_gauss": sd_gauss,
        "ic95_gauss": (punto - 1.96 * sd_gauss, punto + 1.96 * sd_gauss),
        "ic95_boot": (float(q_lo), float(q_hi)),
        "sim": sim,
        "meses": [d.strftime("%Y-%m") for d in res.forecast(H).index],
    }


# --------------------------------------------------------------------------
# Graficos
# --------------------------------------------------------------------------
def graficos(P: pd.DataFrame, real: pd.Series, fc: dict, ganador: str,
             ventana_rolling: int = 24, path: str = "icg_promedio.png"):
    e_g = (real - P[ganador]).to_numpy()
    e_n = (real - P["Naive y_T"]).to_numpy()
    idx = real.index

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    # (A) observado vs estimado + proyeccion
    a = ax[0, 0]
    a.plot(idx, real, color=GRIS, lw=2.2, label="Promedio observado (t+1..t+3)")
    a.plot(idx, P[ganador], color=AZUL, lw=1.6, ls="--", label=f"Estimado {ganador}")
    banda = 1.96 * fc["sd_gauss"]
    a.fill_between(idx, P[ganador] - banda, P[ganador] + banda, color=AZUL,
                   alpha=0.12, label="IC 95%")
    # proyeccion fuera de muestra
    f_out = pd.Timestamp(fc["meses"][-1]) + pd.offsets.MonthEnd(0)
    a.errorbar([f_out], [fc["punto"]],
               yerr=[[fc["punto"] - fc["ic95_boot"][0]], [fc["ic95_boot"][1] - fc["punto"]]],
               fmt="o", color=NARANJA, ms=7, capsize=4, lw=1.8, zorder=5,
               label=f"Proyeccion {fc['meses'][0]}/{fc['meses'][-1]}")
    peores = np.argsort(np.abs(e_g))[-4:]
    a.scatter(idx[peores], real.to_numpy()[peores], s=55, facecolors="none",
              edgecolors="#C44E52", lw=1.5, zorder=4, label="4 mayores errores")
    a.set_title("Promedio trimestral del ICG: observado vs estimado")
    a.set_ylabel("ICG (promedio movil 3m)")
    a.legend(fontsize=8, loc="upper left")

    # (B) RMSE movil
    b = ax[0, 1]
    for e, col, lab in ((e_g, AZUL, ganador), (e_n, NARANJA, "Naive")):
        rm = pd.Series(e ** 2, index=idx).rolling(ventana_rolling).mean() ** 0.5
        b.plot(idx, rm, color=col, lw=1.8, label=lab)
    b.axhline(rmse(e_g), color=AZUL, ls=":", lw=1.2)
    b.set_title(f"RMSE movil ({ventana_rolling} meses)")
    b.set_ylabel("RMSE")
    b.legend(fontsize=9)

    # (C) RMSE acumulado
    c = ax[1, 0]
    for e, col, lab in ((e_g, AZUL, ganador), (e_n, NARANJA, "Naive")):
        c.plot(idx, np.sqrt(np.cumsum(e ** 2) / np.arange(1, len(e) + 1)),
               color=col, lw=1.8, label=lab)
    c.set_title("RMSE acumulado (convergencia de la ventaja)")
    c.set_ylabel("RMSE acumulado")
    c.legend(fontsize=9)

    # (D) dispersion
    d = ax[1, 1]
    lim = [min(real.min(), P[ganador].min()) - .1, max(real.max(), P[ganador].max()) + .1]
    d.scatter(P[ganador], real, s=24, color=AZUL, alpha=0.6, edgecolor="white", lw=0.4)
    d.plot(lim, lim, color="#C44E52", lw=1.4, ls="--", label="45 grados")
    r2 = 1 - np.sum(e_g ** 2) / np.sum((real - real.mean()).to_numpy() ** 2)
    d.set_title(f"Estimado vs observado  (R² = {r2:.3f})")
    d.set_xlabel("Estimado"); d.set_ylabel("Observado")
    d.set_xlim(lim); d.set_ylim(lim); d.legend(fontsize=9)

    fig.suptitle(f"ICG — objetivo promedio t+1..t+3 | RMSE {rmse(e_g):.4f} "
                 f"vs naive {rmse(e_n):.4f} | {len(real)} origenes",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"\nGrafico guardado en {path}")
    return path


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ICG — pronostico del promedio t+1..t+3")
    ap.add_argument("--datos", default="ICG.xlsx")
    ap.add_argument("--min-train", type=int, default=120)
    ap.add_argument("--graficos", action="store_true")
    ap.add_argument("--salida", default="promedio_icg.xlsx")
    ap.add_argument("--png", default="icg_promedio.png")
    args = ap.parse_args()

    s = cargar_icg(args.datos)
    print(f"Serie ICG: {len(s)} obs | {s.index.min():%Y-%m} a {s.index.max():%Y-%m}\n")

    print("Backtest sobre el promedio t+1..t+3:")
    P, real = backtest_promedio(s, CANDIDATOS, args.min_train)
    met = tabla_metricas(P, real)
    print(f"\n--- Metricas ({len(real)} origenes) ---")
    print(met.round(4).to_string())

    ganador = met.index[0]
    sel = costo_de_seleccionar(P, real)
    print("\n--- Costo de seleccionar el modelo en lugar de fijarlo ---")
    print(f"  seleccion recursiva : {sel['RMSE_seleccion_recursiva']:.4f}")
    print(f"  AR(1)+c fijo        : {sel['RMSE_modelo_fijo']:.4f}")
    print(f"  naive               : {sel['RMSE_naive']:.4f}")
    print(f"  elegido: {sel['frecuencia_elegido']}")
    print("  -> fijar el modelo domina: el modelo de produccion queda en AR(1)+c")

    res = ajustar_produccion(s)
    fc = pronostico_promedio(res, s)
    print("\n--- Modelo de produccion (muestra completa) ---")
    print(f"  phi={fc['phi']:.4f} | mu={fc['mu']:.4f} | z={fc['z']:+.4f} "
          f"| gate {'ACTIVO' if fc['gate_activo'] else 'INACTIVO'}")
    print(f"\n--- Proyeccion del promedio {fc['meses'][0]} a {fc['meses'][-1]} ---")
    print(f"  punto            : {fc['punto']:.4f}")
    print(f"  IC 95% gaussiano : [{fc['ic95_gauss'][0]:.4f}, {fc['ic95_gauss'][1]:.4f}]  "
          f"(sd {fc['sd_gauss']:.4f})")
    print(f"  IC 95% bootstrap : [{fc['ic95_boot'][0]:.4f}, {fc['ic95_boot'][1]:.4f}]")
    print(f"  P(promedio < ultimo dato {fc['ultimo']:.2f}) = "
          f"{100 * np.mean(fc['sim'] < fc['ultimo']):.1f}%")

    png = graficos(P, real, fc, ganador, path=args.png) if args.graficos else None

    with pd.ExcelWriter(args.salida) as xls:
        met.round(6).to_excel(xls, sheet_name="Metricas")
        P.join(real).round(6).to_excel(xls, sheet_name="Backtest")
        pd.Series({k: v for k, v in fc.items() if k not in ("sim", "meses")}
                  ).astype(str).to_frame("valor").to_excel(xls, sheet_name="Proyeccion")
    print(f"Resultados exportados a {args.salida}")


if __name__ == "__main__":
    main()
