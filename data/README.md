# Datos

- `paysim_muestra_2pct.csv`: muestra aleatoria del 2 % del dataset PaySim (127.252 transacciones, `random_state=7`).
- `paysim_fraudes.csv`: todas las transacciones fraudulentas (8.213).

Ambos se generaron desde el CSV original de Kaggle (https://www.kaggle.com/datasets/ealaxi/paysim1, licencia CC BY-SA 4.0) con:

```python
import pandas as pd
ruta = "PS_20174392719_1491204439457_log.csv"
pd.concat(c.sample(frac=0.02, random_state=7) for c in pd.read_csv(ruta, chunksize=500_000)).to_csv("paysim_muestra_2pct.csv", index=False)
pd.concat(c[c.isFraud == 1] for c in pd.read_csv(ruta, chunksize=500_000)).to_csv("paysim_fraudes.csv", index=False)
```
