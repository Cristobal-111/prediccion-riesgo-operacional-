"""
Clasificacion de Riesgo Operacional (Bajo / Medio / Alto)
============================================================
Autor: Cristobal Rojas Troncoso

Segunda etapa del proyecto de prediccion de riesgo/discrepancia
operacional: convierte la discrepancia de conciliacion (variable
continua, ver discrepancia_riesgo_operacional.py) en una categoria
de riesgo accionable para negocio, y entrena un clasificador que
predice esa categoria usando solo variables conocidas antes de
ejecutar la transaccion.

Dataset: PaySim - https://www.kaggle.com/datasets/ealaxi/paysim1
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------------------
# 1. Carga y variable objetivo (discrepancia) - igual que en la etapa 1
# ---------------------------------------------------------------------------
DATA_PATH = "PS_20174392719_1491204439457_log.csv"
df = pd.read_csv(DATA_PATH)

sale_dinero = df['type'].isin(['CASH_OUT', 'PAYMENT', 'DEBIT', 'TRANSFER'])
expected_balance = np.where(
    sale_dinero,
    df['oldbalanceOrg'] - df['amount'],
    df['oldbalanceOrg'] + df['amount']
)
df['discrepancy'] = (df['newbalanceOrig'] - expected_balance).abs()

# ---------------------------------------------------------------------------
# 2. Categorizacion del riesgo
# ---------------------------------------------------------------------------
# Bajo: sin discrepancia (conciliacion perfecta)
# Medio/Alto: se dividen en la mediana de las discrepancias distintas de cero,
# generando 3 categorias razonablemente balanceadas para negocio.
nonzero_median = df.loc[df['discrepancy'] > 0, 'discrepancy'].median()

def categorize(d):
    if d == 0:
        return 'Bajo'
    elif d <= nonzero_median:
        return 'Medio'
    else:
        return 'Alto'

df['risk_category'] = df['discrepancy'].apply(categorize)
print("Distribucion de categorias de riesgo:")
print(df['risk_category'].value_counts(normalize=True) * 100)

# ---------------------------------------------------------------------------
# 3. Features (mismas que en la etapa 1, sin leakage)
# ---------------------------------------------------------------------------
features_df = pd.get_dummies(
    df[['type', 'amount', 'step', 'oldbalanceOrg', 'oldbalanceDest']],
    columns=['type'], drop_first=True
)
X = features_df
y = df['risk_category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------------
# 4. Modelo 1: Regresion Logistica (interpretable, punto de partida)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)
y_pred_lr = logreg.predict(X_test_scaled)

print("\n=== Regresion Logistica ===")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# ---------------------------------------------------------------------------
# 5. Modelo 2: Random Forest (mejor desempeño, usado en el dashboard)
# ---------------------------------------------------------------------------
# Nota: para datasets muy grandes, se recomienda entrenar sobre una muestra
# (aqui 300k filas) para mantener tiempos de entrenamiento razonables.
sample_idx = np.random.RandomState(42).choice(len(X_train), min(300_000, len(X_train)), replace=False)
X_train_sample = X_train.iloc[sample_idx]
y_train_sample = y_train.iloc[sample_idx]

rf = RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_sample, y_train_sample)
y_pred_rf = rf.predict(X_test)

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\nImportancia de variables (Random Forest):")
print(importances)

# ---------------------------------------------------------------------------
# 6. Exportar modelo y artefactos para el dashboard
# ---------------------------------------------------------------------------
with open('rf_model.pkl', 'wb') as f:
    pickle.dump({'model': rf, 'columns': list(X.columns)}, f)

importances.to_csv('importancias.csv', index=False)

# ---------------------------------------------------------------------------
# 7. Conclusiones
# ---------------------------------------------------------------------------
print("""
CONCLUSIONES
============
1. El Random Forest supera claramente a la regresion logistica
   (88% vs 78% de accuracy), especialmente en distinguir Bajo de Medio.
2. La categoria "Alto" se predice casi a la perfeccion (F1 ~ 0.99) en
   ambos modelos, lo que la hace confiable para priorizar revisiones.
3. A diferencia del modelo de regresion (donde 'amount' dominaba), en
   la clasificacion 'oldbalanceOrg' y 'amount' concentran juntos ~61%
   de la importancia -> el riesgo depende tanto del monto como del
   contexto de saldo previo del cliente.
4. Aplicacion practica: el modelo puede anotar cada transaccion nueva
   con una probabilidad de riesgo Alto ANTES de que se ejecute o
   concilie, permitiendo priorizar la revision manual del equipo de
   control de gestion (ver dashboard/ para el panel interactivo).
""")
