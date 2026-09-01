# Predicción de Riesgo/Discrepancia Operacional

Modelo de regresión lineal que predice la **magnitud de una discrepancia de conciliación** en transacciones financieras, en función de variables del proceso (tipo de transacción, monto, saldos previos, antigüedad).

Proyecto basado en la metodología del **Google Advanced Data Analytics Professional Certificate** (Coursera), replicando el enfoque usado para predecir `fare_amount` en un dataset de taxis, pero aplicado a un caso de **riesgo/control de gestión operacional**.

## Dataset

**PaySim** — Synthetic Financial Datasets For Fraud Detection
Fuente: https://www.kaggle.com/datasets/ealaxi/paysim1
Simulador de transacciones de dinero móvil (6.3M+ registros) basado en logs reales de un servicio financiero.

> Lopez-Rojas, E. A., Elmir, A., & Axelsson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection*. The 28th European Modeling and Simulation Symposium (EMSS), Larnaca, Cyprus.

## Metodología

### 1. Ingeniería de la variable objetivo

PaySim no trae una columna de "discrepancia" explícita, así que se construyó una:

```
discrepancia = |saldo_reportado_post_transacción − saldo_esperado_post_transacción|
```

El saldo esperado se calcula según el tipo de transacción (el dinero sale o entra de la cuenta origen). Esta métrica es directamente análoga a lo que un analista revisa al conciliar movimientos contra saldos esperados.

### 2. Features utilizadas

Solo variables conocidas **antes o al momento** de la transacción (se excluyen los saldos posteriores para evitar *data leakage*, ya que se usaron para construir el target):

- `type` (codificada como dummies)
- `amount`
- `step` (antigüedad temporal)
- `oldbalanceOrg`
- `oldbalanceDest`

### 3. Modelo

`LinearRegression` de scikit-learn, sobre features estandarizadas con `StandardScaler`, split 80/20.

## Resultados

| Métrica | Valor |
|---|---|
| R² | 0.963 |
| MAE | 41,798 |

### Coeficientes (efecto estandarizado sobre la discrepancia)

| Feature | Coeficiente |
|---|---|
| `amount` | **572,937** |
| `type_PAYMENT` | 72,684 |
| `type_CASH_OUT` | 64,882 |
| `type_TRANSFER` | 42,300 |
| `type_DEBIT` | 12,305 |
| `oldbalanceDest` | 7,457 |
| `oldbalanceOrg` | −4,035 |
| `step` | −1,091 |

**Feature con mayor efecto: `amount`**

Como se usó `StandardScaler`, cada coeficiente representa el cambio esperado en la discrepancia (en unidades monetarias) asociado a un aumento de una desviación estándar en esa variable, manteniendo las demás constantes.

## Conclusiones

1. El **monto de la transacción** es, por lejos, la variable con mayor efecto sobre la magnitud de la discrepancia de conciliación, muy por encima del tipo de transacción o los saldos previos.
2. Las transacciones tipo `PAYMENT`, `CASH_OUT` y `TRANSFER` concentran el mayor riesgo de discrepancia frente a `DEBIT` y `CASH_IN`.
3. Un modelo lineal simple ya explica ~96% de la variabilidad (R² ≈ 0.96) — un buen punto de partida para un sistema de alerta temprana sin necesidad de modelos más complejos.
4. **Aplicación práctica:** estos resultados permiten priorizar la revisión manual de conciliaciones hacia transacciones de alto monto y de los tipos con mayor coeficiente, optimizando el tiempo del equipo de control de gestión.

## Etapa 2 — Clasificación de riesgo (Bajo / Medio / Alto)

La discrepancia continua se convierte en una categoría de riesgo accionable para negocio:

- **Bajo**: sin discrepancia (conciliación perfecta) — 30.5%
- **Medio**: discrepancia entre 0 y la mediana de los casos con discrepancia — 34.7%
- **Alto**: discrepancia sobre esa mediana — 34.7%

Se compararon dos modelos, usando las mismas features (sin leakage) que en la etapa 1:

| Modelo | Accuracy | Notas |
|---|---|---|
| Regresión Logística | 78% | Interpretable, buen baseline |
| **Random Forest** | **88%** | Mejor desempeño, usado en el dashboard |

En ambos modelos, la categoría **"Alto"** se predice casi a la perfección (F1 ≈ 0.99), lo que la hace confiable para priorizar revisiones. En el Random Forest, `oldbalanceOrg` (33%) y `amount` (29%) concentran la mayor importancia — a diferencia de la regresión, donde `amount` dominaba solo.

Script: [`clasificacion_riesgo.py`](clasificacion_riesgo.py)

## Etapa 3 — Dashboard interactivo (Streamlit)

Panel de riesgo operacional en `dashboard/dashboard.py`, con:

- KPIs: transacciones filtradas, % riesgo Alto, monto expuesto, precisión del modelo
- Filtros por tipo de transacción, rango de monto y categoría de riesgo
- Ranking de las transacciones con mayor probabilidad de riesgo Alto (para priorizar revisión manual)
- Gráfico de importancia de variables y matriz de confusión en vivo

```bash
cd dashboard
pip install -r requirements.txt
streamlit run dashboard.py
```

El dashboard usa una muestra de 15.000 transacciones de validación (`dashboard_sample.csv`) con las predicciones ya calculadas, para que corra liviano sin necesitar el CSV completo de 471 MB. El modelo entrenado (`rf_model.pkl`) queda disponible para anotar nuevas transacciones.

## Cómo ejecutar el proyecto completo

```bash
pip install pandas numpy scikit-learn matplotlib streamlit
python discrepancia_riesgo_operacional.py   # Etapa 1: regresión
python clasificacion_riesgo.py              # Etapa 2: clasificación
cd dashboard && streamlit run dashboard.py  # Etapa 3: dashboard
```

(Descargar el CSV de Kaggle y ajustar `DATA_PATH` en los scripts.)

## Stack

Python · pandas · scikit-learn · matplotlib · Streamlit
