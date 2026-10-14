# Versión 1 (original)

Scripts, dashboard y artefactos de la primera versión, conservados tal como se publicaron.
Los scripts esperan el CSV completo de PaySim (`PS_20174392719_1491204439457_log.csv`, Kaggle) en esta carpeta.

Lo que la versión 2 encontró:
1. La "discrepancia de conciliación" es exactamente `máx(0, monto − saldo previo)`: la regresión (R² 0,96) aproximaba una resta entre sus propias variables de entrada.
2. Las categorías Bajo / Medio / Alto se reproducen al 100 % con una regla de una línea (el Random Forest lograba 88 %). Además, ~13 % de las transacciones quedaba como riesgo "Medio" por diferencias de 1 centavo de redondeo.
3. La discrepancia refleja transacciones ejecutadas sin saldo suficiente (73 % de las salidas): una simplificación del simulador, no un riesgo de conciliación.

Ver `../notebooks/riesgo_operacional_paysim_v2.ipynb`.
