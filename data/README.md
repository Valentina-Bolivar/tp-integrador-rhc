# Datos: Right Heart Catheterization (RHC)

**Fuente:** Connors, A. F., Speroff, T., Dawson, N. V., et al. (1996). The effectiveness of right heart catheterization in the initial care of critically ill patients. *JAMA, 276*(11), 889–897.
**Descarga:** <https://hbiostat.org/data/repo/rhc.csv> (Departamento de Bioestadística, Vanderbilt University).
**Origen:** estudio observacional SUPPORT, pacientes adultos críticos en cinco hospitales universitarios de EE. UU. (1989 a 1994).

El archivo `raw/rhc.csv` es la copia sin modificar. Si se borra, `rhc.data.load_raw()` lo descarga nuevamente.

## Dimensiones

- **5.735 pacientes**, 63 columnas originales.
- **2.184 tratados** (RHC en las primeras 24 h) y 3.551 controles.

## Roles de las variables

| Rol | Variable | Descripción |
|---|---|---|
| Tratamiento (T) | `swang1` (recodificada como `rhc`) | 1 = cateterismo cardíaco derecho dentro de las 24 h de la admisión |
| Resultado principal (Y) | `dth30` | 1 = muerte dentro de los 30 días |
| Resultado secundario | `death` | 1 = muerte dentro de los 180 días |
| Resultado continuo (Fase 2) | `los` (derivada) | Días de internación = alta − admisión (1 caso sin fecha de alta queda como faltante) |
| Post-tratamiento (excluidas) | `sadmdte`, `dschdte`, `dthdte`, `lstctdte`, `t3d30` | Fechas y tiempos medidos después de la decisión de tratar |
| Identificadores (excluidas) | `ptid`, columna índice | Sin contenido informativo |

## Diccionario de covariables basales (Z)

| Grupo | Variables |
|---|---|
| Demografía y nivel socioeconómico | `age`, `sex`, `race`, `edu` (años de educación), `income` (4 tramos), `ninsclas` (cobertura médica) |
| Diagnóstico de ingreso | `cat1` (categoría primaria, 9 niveles), `cat2` (categoría secundaria), `ca` (cáncer: no / sí / metastásico), `resp` … `ortho` (10 indicadores de categoría de admisión) |
| Comorbilidades (0/1) | `cardiohx`, `chfhx`, `dementhx`, `psychhx`, `chrpulhx`, `renalhx`, `liverhx`, `gibledhx`, `malighx`, `immunhx`, `transhx`, `amihx` |
| Gravedad y pronóstico | `aps1` (APACHE III), `scoma1` (escala de coma SUPPORT; 0 = sin coma), `surv2md1` (probabilidad de sobrevida a 2 meses estimada por el modelo SUPPORT), `das2d3pc` (índice de capacidad funcional DASI), `dnr1` (orden de no reanimar en el día 1) |
| Fisiología del día 1 | `meanbp1`, `wblc1`, `hrt1`, `resp1`, `temp1`, `pafi1`, `alb1`, `hema1`, `bili1`, `crea1`, `sod1`, `pot1`, `paco21`, `ph1`, `wtkilo1` |
| Con faltantes masivos | `adld3p` (índice ADL, 75 % faltante), `urin1` (diuresis, 53 % faltante) |

Las descripciones se basan en la documentación de SUPPORT publicada por Vanderbilt (<https://biostat.app.vumc.org/wiki/pub/Main/DataSets/rhc.html>).
