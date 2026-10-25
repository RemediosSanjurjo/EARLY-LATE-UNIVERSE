# TFM – Modelo Early/Late Dark Energy con curvatura

Código del TFM: ajuste del modelo de energía oscura Early/Late con curvatura
espacial a los datos de Planck 2018 usando CAMB y Cobaya.

## Contenido

- `early_late_theory_curvatura.py`: clase `Theory` de Cobaya que calcula los C_l con CAMB.
- `funciones.py`: evaluación y minimización de chi², cálculo de C_l y de E²(a).

## Instalación

```bash
pip install -r requirements.txt
```

Los datos de Planck (likelihood `COM_Likelihood_Data-baseline_R3.00.tar.gz` y
espectros `COM_PowerSpect_CMB-*.txt`) **no** están en el repositorio; se descargan
del Planck Legacy Archive.

## Uso mínimo

```python
import sys; sys.path.append("ruta/a/este/repo")
import funciones as f

f.set_likelihood_paths("/ruta/a/planck_2018")
chi2, partes, tabla = f.evaluar_modelo_early_late_curvatura(
    H0=73.5, Omega_E0=1e-4, w_E=0.2, Omega_k0=0.0, likelihood="lite")
```
