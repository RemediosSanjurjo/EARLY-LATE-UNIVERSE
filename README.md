# TFM – Modelo Early/Late Dark Energy con curvatura

Modelo de energía oscura Early/Late con curvatura
espacial. Se evalúan los datos de Planck 2018 mediante CAMB y Cobaya.

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

# Uso mínimo del código

Este apartado muestra cómo utilizar las funciones principales del modelo cosmológico **EARLY-LATE**, tanto para evaluar el ajuste de un conjunto de parámetros como para ejecutar un análisis mediante Monte Carlo Markov Chain (MCMC).

## 1. Evaluación del ajuste mediante mínimos cuadrados

La función `evaluar_modelo_early_late_curvatura` permite evaluar el ajuste de un conjunto dado de parámetros cosmológicos mediante la función \(\chi^2\).

```python
chi2_i, partes_i, sample_i = evaluar_modelo_early_late_curvatura(
    H0=72,
    Omega_E0=0.01,
    w_E=0.2,
    Omega_k0=0,
    likelihood=lite,
    output=None,
    debug=False
)
```

La función devuelve:

- `chi2_i`: valor total de la función \(\chi^2\).
- `partes_i`: contribuciones individuales de las distintas likelihoods utilizadas, cuando se emplea una combinación de ellas.

### Requisito para utilizar Planck Lite

Para utilizar la likelihood `lite`, es necesario descargar los datos correspondientes desde [Planck Legacy Archive](https://pla.esac.esa.int/).

En particular, se necesita el paquete `COM_Likelihood_Data-baseline_R3.00.tar.gz` y el archivo de likelihood `plik_lite_v22_TTTEEE.clik`.

La ruta al archivo debe configurarse en el entorno de ejecución antes de evaluar el modelo.

## 2. Ejecución de Monte Carlo Markov Chain (MCMC)

El siguiente ejemplo muestra cómo configurar y ejecutar un análisis MCMC con el modelo EARLY-LATE.

En esta configuración se mantienen fijos algunos parámetros cosmológicos, mientras que otros se consideran libres y se muestrean dentro de los intervalos definidos por sus priors.

```python
info_MCMC_w025_noc = {
    "theory": {
        "early_late_theory_curvatura.EarlyLateTheoryCurvatura": {
            "lmax": 2508,
            "TCMB": TCMB,
            "a_min": 1e-5,
            "n_a": 5000,
            "stop_at_error": True
        }
    },

    "likelihood": {
        "planck_2018_highl_plik.TTTEEE_lite": {
            "clik_file": plik_lite_file
        }
    },

    "params": {
        # Parámetros fijos
        "H0": 73.5,
        "As": 2.1e-9,
        "mnu": 0.06,
        "nnu": 3.046,
        "w_E": 0.2526,
        "Omega_k0": 0,
        "tau": 0.0544,
        "Omega_E0": 0.000048,

        # Parámetros libres
        "ns": {
            "prior": {"min": 0.95, "max": 0.99},
            "ref": 0.965
        },

        "omch2": {
            "prior": {"min": 0.10, "max": 0.13},
            "ref": 0.12
        },

        "ombh2": {
            "prior": {"min": 0.021, "max": 0.025},
            "ref": 0.0225
        },

        # Parámetros derivados
        "Omega_m0_derived": {"derived": True},
        "Omega_r0_derived": {"derived": True},
        "Omega_L0_derived": {"derived": True},

        # Parámetro de calibración de Planck Lite
        "A_planck": 1.0
    },

    "sampler": {
        "mcmc": {
            "max_tries": 100000,
            "Rminus1_stop": 0.02,
            "Rminus1_cl_stop": 0.3,
            "covmat": "auto"
        }
    },

    "output": "MCMC_EarlyLate_w025_noc",
    "debug": False
}

updated_info_w025_noc, sampler_w025_noc = run(
    info_MCMC_w025_noc,
    force=True
)
```

### Descripción de la configuración

- **Teoría:** define los parámetros numéricos y las condiciones de cálculo del modelo.
- **Likelihood:** especifica la likelihood de Planck Lite y la ubicación del archivo necesario.
- **Parámetros fijos:** establece los valores cosmológicos que no se muestrean.
- **Parámetros libres:** define los priors y los valores de referencia de los parámetros que se muestrean.
- **Parámetros derivados:** solicita el cálculo de cantidades cosmológicas adicionales.
- **Sampler:** establece los criterios de convergencia y las opciones de muestreo MCMC.
- **Output:** especifica el prefijo o directorio utilizado para guardar los resultados, según la configuración de ejecución.
