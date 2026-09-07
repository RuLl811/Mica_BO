"""
=============================================================================
 ICG — Modelo de produccion: ARIMA(1,0,0) con constante, proyeccion a 3 meses
=============================================================================
Modelo ganador del backtest de horizonte fijo t+3 (ver icg_backtest.py).

    y_t - mu = phi * (y_{t-1} - mu) + eps_t,     eps_t ~ WN(0, sigma^2)

Proyeccion a h pasos:  E[y_{T+h}] = mu + phi^h * (y_T - mu)
Varianza a h pasos:    sigma^2 * (1 - phi^{2h}) / (1 - phi^2)

Entrega:
  * proyeccion puntual h = 1, 2, 3
  * IC 95% parametrico (gaussiano, del filtro de Kalman)
  * IC 95% empirico, construido con los cuantiles de los errores t+3
    efectivamente observados en el backtest — control de robustez frente
    a la no normalidad de las innovaciones
  * diagnostico completo de residuos y cobertura historica de los IC

Uso:
    python icg_produccion.py --datos ICG.xlsx --h 3 --salida proyeccion_icg.xlsx
=============================================================================
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

from icg_backtest import cargar_icg

warnings.filterwarnings("ignore")

ORDEN = (1, 0, 0)
TREND = "c"


# --------------------------------------------------------------------------
def ajustar(serie: pd.Series):
    """Estima el ARIMA(1,0,0)+c sobre la muestra completa."""
    return SARIMAX(serie, order=ORDEN, trend=TREND,
                   enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)


def parametros_estructurales(res) -> dict:
    """Traduce los coeficientes a magnitudes economicamente interpretables."""
    c = float(res.params.iloc[0])
    phi = float(res.params.iloc[1])
    sigma2 = float(res.params.iloc[-1])
    return {
        "phi": phi,
        "constante": c,
        "mu_largo_plazo": c / (1 - phi),
        "vida_media_meses": np.log(0.5) / np.log(phi),
        "sigma_innovacion": np.sqrt(sigma2),
        "desvio_incondicional": np.sqrt(sigma2 / (1 - phi ** 2)),
        "peso_reversion_h3": 1 - phi ** 3,   # cuanto pesa mu en la proyeccion t+3
    }


def diagnostico_residuos(res) -> pd.DataFrame:
    """Ljung-Box (autocorrelacion), Ljung-Box^2 (ARCH) y Jarque-Bera."""
    r = res.resid[1:]
    lb = acorr_ljungbox(r, lags=[3, 6, 12, 24], return_df=True)
    lb2 = acorr_ljungbox(r ** 2, lags=[6, 12], return_df=True)
    jb, jb_p = stats.jarque_bera(r)[:2]
    filas = [{"prueba": f"Ljung-Box({k})", "estadistico": v.lb_stat, "p_valor": v.lb_pvalue}
             for k, v in lb.iterrows()]
    filas += [{"prueba": f"Ljung-Box^2({k}) [ARCH]", "estadistico": v.lb_stat,
               "p_valor": v.lb_pvalue} for k, v in lb2.iterrows()]
    filas += [{"prueba": "Jarque-Bera", "estadistico": jb, "p_valor": jb_p},
              {"prueba": "curtosis", "estadistico": stats.kurtosis(r), "p_valor": np.nan},
              {"prueba": "asimetria", "estadistico": stats.skew(r), "p_valor": np.nan}]
    return pd.DataFrame(filas)


def errores_backtest_h(serie: pd.Series, h: int = 3, min_train: int = 120) -> np.ndarray:
    """Errores t+h fuera de muestra del modelo de produccion (para IC empirico)."""
    errs = []
    for t in range(min_train, len(serie) - h + 1):
        r = ajustar(serie.iloc[:t])
        errs.append(float(serie.iloc[t + h - 1]) - float(r.forecast(h).iloc[-1]))
    return np.asarray(errs)


def cobertura_historica(serie: pd.Series, h: int = 3, min_train: int = 120,
                        alphas=(0.20, 0.10, 0.05)) -> pd.DataFrame:
    """Cobertura empirica de los IC nominales en el horizonte objetivo.

    Un IC 95% que historicamente cubre el 80% no sirve para dimensionar
    riesgo; esta tabla es el control de calibracion del modelo.
    """
    dentro = {a: [] for a in alphas}
    for t in range(min_train, len(serie) - h + 1):
        r = ajustar(serie.iloc[:t])
        fc = r.get_forecast(h)
        real = float(serie.iloc[t + h - 1])
        for a in alphas:
            lo, hi = fc.conf_int(alpha=a).iloc[-1].to_numpy()
            dentro[a].append(lo <= real <= hi)
    return pd.DataFrame([{"nominal_%": 100 * (1 - a),
                          "empirica_%": 100 * np.mean(v),
                          "n": len(v)} for a, v in dentro.items()])


def proyectar(res, serie: pd.Series, h: int, errores_h: np.ndarray | None) -> pd.DataFrame:
    """Proyeccion puntual + IC 95% parametrico y (opcional) empirico."""
    fc = res.get_forecast(h)
    ci = fc.conf_int(alpha=0.05)
    tab = pd.DataFrame({
        "h": np.arange(1, h + 1),
        "pronostico": fc.predicted_mean.to_numpy(),
        "ic95_inf": ci.iloc[:, 0].to_numpy(),
        "ic95_sup": ci.iloc[:, 1].to_numpy(),
        "ee_pronostico": fc.se_mean.to_numpy(),
    }, index=fc.predicted_mean.index)
    tab.index.name = "fecha"

    if errores_h is not None and len(errores_h) > 30:
        q_lo, q_hi = np.quantile(errores_h, [0.025, 0.975])
        tab["ic95_emp_inf"] = np.nan
        tab["ic95_emp_sup"] = np.nan
        tab.iloc[-1, tab.columns.get_loc("ic95_emp_inf")] = tab["pronostico"].iloc[-1] + q_lo
        tab.iloc[-1, tab.columns.get_loc("ic95_emp_sup")] = tab["pronostico"].iloc[-1] + q_hi

    # probabilidades utiles a t+h sobre el ultimo dato observado
    ult = float(serie.iloc[-1])
    z = (ult - tab["pronostico"]) / tab["ee_pronostico"]
    tab["P(ICG > ultimo obs)"] = 1 - stats.norm.cdf(z)
    return tab


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Proyeccion de produccion del ICG")
    ap.add_argument("--datos", default="ICG.xlsx")
    ap.add_argument("--h", type=int, default=3)
    ap.add_argument("--min-train", type=int, default=120)
    ap.add_argument("--sin-calibracion", action="store_true",
                    help="omite el recalculo de cobertura e IC empirico (mas rapido)")
    ap.add_argument("--salida", default="proyeccion_icg.xlsx")
    args = ap.parse_args()

    s = cargar_icg(args.datos)
    print(f"Serie ICG: {len(s)} obs | {s.index.min():%Y-%m} a {s.index.max():%Y-%m} | "
          f"ultimo valor {s.iloc[-1]:.3f}\n")

    res = ajustar(s)
    print(res.summary().tables[1])

    par = parametros_estructurales(res)
    print("\n--- Lectura estructural ---")
    print(f"  phi                      : {par['phi']:.4f}")
    print(f"  mu (media de largo plazo): {par['mu_largo_plazo']:.4f}")
    print(f"  vida media del shock     : {par['vida_media_meses']:.1f} meses")
    print(f"  sigma innovacion         : {par['sigma_innovacion']:.4f}")
    print(f"  peso de mu en t+3        : {100 * par['peso_reversion_h3']:.1f}%")

    print("\n--- Diagnostico de residuos ---")
    print(diagnostico_residuos(res).round(4).to_string(index=False))

    errores, cob = None, None
    if not args.sin_calibracion:
        print("\n--- Calibracion (recursion de origenes) ---")
        cob = cobertura_historica(s, h=args.h, min_train=args.min_train)
        print(cob.round(2).to_string(index=False))
        errores = errores_backtest_h(s, h=args.h, min_train=args.min_train)
        print(f"  RMSE t+{args.h} fuera de muestra: {np.sqrt(np.mean(errores ** 2)):.4f} | "
              f"sesgo: {errores.mean():+.4f}")

    proy = proyectar(res, s, args.h, errores)
    print(f"\n--- Proyeccion ICG a {args.h} meses ---")
    print(proy.round(4).to_string())

    with pd.ExcelWriter(args.salida) as xls:
        proy.round(6).to_excel(xls, sheet_name="Proyeccion")
        pd.Series(par).round(6).to_frame("valor").to_excel(xls, sheet_name="Parametros")
        diagnostico_residuos(res).round(6).to_excel(xls, sheet_name="Diagnostico", index=False)
        if cob is not None:
            cob.round(4).to_excel(xls, sheet_name="Calibracion", index=False)
        s.to_frame("ICG").to_excel(xls, sheet_name="Serie")
    print(f"\nProyeccion exportada a {args.salida}")


if __name__ == "__main__":
    main()
