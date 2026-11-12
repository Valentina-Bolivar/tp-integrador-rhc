# Cateterismo cardíaco derecho: predicción y decisión causal

![Python](https://img.shields.io/badge/python-3.12-3b6e8f)
![Quality Gate](https://github.com/Valentina-Bolivar/tp-integrador-rhc/actions/workflows/quality-gate.yml/badge.svg)
![Licencia de datos](https://img.shields.io/badge/datos-Vanderbilt%20Biostatistics-8fb3c9)

**Trabajo Práctico Integrador. Estadística Aplicada y Modelos de Regresión (EAyMR 2026), UCA.**

Este proyecto evalúa si el cateterismo cardíaco derecho (RHC) aplicado en las primeras 24 horas de internación modifica la mortalidad a 30 días de pacientes críticos. Distingue dos preguntas:

| Pregunta | Formalización | Fase |
|---|---|---|
| ¿Qué paciente tiene mayor riesgo de morir? | P(Y \| X) | Fase 3: torneo predictivo |
| ¿Qué pasaría si se le aplica RHC? | P(Y \| do(T = 1)) | Fase 4: inferencia causal |

**App en producción:** pendiente (se completa en el Paso 5).
**Notebook:** [`notebooks/TP_Integrador_Final.ipynb`](notebooks/TP_Integrador_Final.ipynb) · [Abrir en Colab](https://colab.research.google.com/github/Valentina-Bolivar/tp-integrador-rhc/blob/main/notebooks/TP_Integrador_Final.ipynb)
**Bitácora:** [`BITACORA.md`](BITACORA.md)

## Equipo y roles Scrum

| Rol | Responsable |
|---|---|
| Project Lead & Scrum Master | Valentina Bolívar |
| Backend & Machine Learning / Causal Engineer | Valentina Bolívar |
| Frontend & UI/UX Engineer | Felicitas Paz |
| QA & Validation Engineer | Emilia Bedners |

**Regla de integración:** ningún pull request se incorpora a `main` sin el veredicto **[APROBADO]** del QA Engineer y la verificación automática del Quality Gate.

## Estado del proyecto

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Estructura, contrato de datos, Quality Gate | Completa |
| 1 | Auditoría forense, EDA, contrastes de hipótesis, ANOVA | Completa |
| 2 | MCO, diagnóstico de Gauss-Markov, Cook, VIF, PCA sin fuga | Completa |
| 3 | Torneo de ML y calibración | Pendiente |
| 4 | DAG, DoWhy, EconML, refutaciones, ensayo crítico | Pendiente |
| 5 | Web app desplegada | Pendiente |

## Estructura del repositorio

```
├── app/                  Web app en Streamlit
├── data/                 Dataset original y diccionario de variables
├── models/               Modelos serializados (joblib / json)
├── notebooks/            Notebook maestro reproducible
├── reports/figures/      Figuras exportadas
├── src/rhc/              Código modular reutilizado por el notebook y la app
├── tests/                Tests automáticos (pytest)
├── .github/workflows/    Quality Gate (ruff + pytest en cada pull request)
├── BITACORA.md           Bitácora de ingeniería
├── requirements.txt      Dependencias fijadas
└── requirements-dev.txt  Herramientas de calidad
```

## Reproducción

**En Google Colab:** abrir el notebook con el enlace "Abrir en Colab" y elegir la opción *Ejecutar todas* del menú *Entorno de ejecución*. La primera celda descarga el repositorio dentro de Colab (el notebook necesita el código de `src/rhc` y el dataset de `data/raw`) e instala las versiones fijadas en `requirements.txt`. Si Colab solicita reiniciar la sesión después de la instalación, se reinicia y se vuelven a ejecutar todas las celdas.

**En una computadora local (Python 3.12):**

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest                           # Quality Gate local
ruff check src tests app
jupyter nbconvert --execute --to notebook --inplace notebooks/TP_Integrador_Final.ipynb
streamlit run app/app.py
```

Semilla global: `SEED = 42` (definida en `src/rhc/config.py`).

## Datos

Connors, A. F., et al. (1996). *JAMA, 276*(11), 889–897. Detalle y diccionario en [`data/README.md`](data/README.md).
