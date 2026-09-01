"""
Dashboard de Riesgo/Discrepancia Operacional
==============================================
Autor: Cristobal Rojas Troncoso

Visualiza las predicciones del modelo de clasificacion de riesgo
(Bajo/Medio/Alto) entrenado sobre el dataset PaySim, en el mismo
espiritu que el dashboard de seguimiento desarrollado para KPEX.

Ejecutar con:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Riesgo Operacional - Conciliacion", layout="wide")

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard_sample.csv")
    return df

@st.cache_data
def load_importances():
    return pd.read_csv("importancias.csv")

df = load_data()
importances = load_importances()

RISK_ORDER = ["Bajo", "Medio", "Alto"]
RISK_COLORS = {"Bajo": "#16a34a", "Medio": "#f59e0b", "Alto": "#dc2626"}

# ---------------------------------------------------------------------------
# Sidebar - filtros
# ---------------------------------------------------------------------------
st.sidebar.header("Filtros")

tipos = sorted(df["type"].unique())
tipos_sel = st.sidebar.multiselect("Tipo de transaccion", tipos, default=tipos)

monto_min, monto_max = float(df["amount"].min()), float(df["amount"].max())
rango_monto = st.sidebar.slider(
    "Rango de monto",
    min_value=monto_min,
    max_value=monto_max,
    value=(monto_min, monto_max)
)

riesgo_sel = st.sidebar.multiselect(
    "Categoria de riesgo predicha", RISK_ORDER, default=RISK_ORDER
)

df_filtrado = df[
    (df["type"].isin(tipos_sel))
    & (df["amount"].between(rango_monto[0], rango_monto[1]))
    & (df["risk_predicho"].isin(riesgo_sel))
]

# ---------------------------------------------------------------------------
# Header y KPIs
# ---------------------------------------------------------------------------
st.title("Panel de Riesgo/Discrepancia Operacional")
st.caption(
    "Prediccion de la categoria de riesgo de conciliacion (Bajo / Medio / Alto) "
    "a partir de variables conocidas antes de ejecutar la transaccion. "
    "Dataset: PaySim (muestra de validacion, 15.000 transacciones)."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transacciones (filtro actual)", f"{len(df_filtrado):,}")
pct_alto = (df_filtrado["risk_predicho"] == "Alto").mean() * 100 if len(df_filtrado) else 0
col2.metric("% Riesgo Alto", f"{pct_alto:.1f}%")
monto_alto = df_filtrado.loc[df_filtrado["risk_predicho"] == "Alto", "amount"].sum()
col3.metric("Monto en transacciones de Alto riesgo", f"${monto_alto:,.0f}")
accuracy_local = (df_filtrado["risk_real"] == df_filtrado["risk_predicho"]).mean() * 100 if len(df_filtrado) else 0
col4.metric("Precision del modelo (muestra filtrada)", f"{accuracy_local:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Fila 1: distribucion de riesgo + importancia de variables
# ---------------------------------------------------------------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Distribucion de riesgo por tipo de transaccion")
    tabla = (
        df_filtrado.groupby(["type", "risk_predicho"]).size().unstack(fill_value=0)
    )
    tabla = tabla.reindex(columns=RISK_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    tabla.plot(
        kind="bar", stacked=True, ax=ax,
        color=[RISK_COLORS[r] for r in RISK_ORDER]
    )
    ax.set_xlabel("Tipo de transaccion")
    ax.set_ylabel("N° transacciones")
    ax.legend(title="Riesgo")
    plt.xticks(rotation=30)
    st.pyplot(fig)

with c2:
    st.subheader("Importancia de variables (Random Forest)")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(importances["feature"], importances["importance"], color="#2563eb")
    ax2.invert_yaxis()
    ax2.set_xlabel("Importancia")
    st.pyplot(fig2)

st.divider()

# ---------------------------------------------------------------------------
# Fila 2: ranking de transacciones de mayor riesgo
# ---------------------------------------------------------------------------
st.subheader("Top transacciones a revisar (mayor probabilidad de riesgo Alto)")

top_riesgo = df_filtrado.sort_values("prob_Alto", ascending=False).head(20)
st.dataframe(
    top_riesgo[["type", "amount", "oldbalanceOrg", "oldbalanceDest",
                "risk_predicho", "prob_Alto"]]
    .rename(columns={
        "type": "Tipo", "amount": "Monto",
        "oldbalanceOrg": "Saldo origen (previo)",
        "oldbalanceDest": "Saldo destino (previo)",
        "risk_predicho": "Riesgo predicho",
        "prob_Alto": "Prob. riesgo Alto"
    }),
    use_container_width=True
)

st.divider()

# ---------------------------------------------------------------------------
# Fila 3: desempeno del modelo
# ---------------------------------------------------------------------------
st.subheader("Desempeño del modelo sobre la muestra filtrada")
if len(df_filtrado):
    matriz = pd.crosstab(
        df_filtrado["risk_real"], df_filtrado["risk_predicho"],
        rownames=["Real"], colnames=["Predicho"]
    ).reindex(index=RISK_ORDER, columns=RISK_ORDER, fill_value=0)
    st.dataframe(matriz, use_container_width=True)
else:
    st.info("No hay transacciones con los filtros seleccionados.")

st.caption(
    "Modelo: RandomForestClassifier · Accuracy global en set de test: 88% · "
    "Variables: tipo de transaccion, monto, antiguedad, saldos previos (origen/destino)."
)
