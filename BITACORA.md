# Bitácora de Ingeniería

Cuaderno de laboratorio del equipo: registra las decisiones metodológicas y su justificación. No resume el notebook ni reemplaza el historial de commits.

| Rol Scrum | Responsable |
|---|---|
| Project Lead & Scrum Master | Valentina Bolívar |
| Backend & Machine Learning / Causal Engineer | Valentina Bolívar |
| Frontend & UI/UX Engineer | Felicitas Paz |
| QA & Validation Engineer | Emilia Bedners |

---

## Entrada 1. Marco estratégico e hipótesis

**Fecha:** 12/11/2026 · **Firma:** Valentina Bolívar (Project Lead & Scrum Master)

### Preguntas fundacionales

El proyecto se plantea desde la perspectiva de la dirección médica de una unidad de cuidados intensivos que debe decidir si el cateterismo cardíaco derecho (RHC, o catéter de Swan-Ganz) se incorpora como práctica rutinaria en la evaluación inicial de pacientes críticos. El procedimiento es invasivo y costoso, y su beneficio clínico fue cuestionado desde la publicación de Connors et al. (1996).

1. **Pregunta predictiva**, P(Y | X): dado el perfil clínico de un paciente al ingreso, ¿cuál es su probabilidad de morir dentro de los 30 días?
2. **Pregunta causal**, P(Y | do(T = t)): si la unidad decidiera aplicar RHC a un paciente, ¿cómo cambiaría su probabilidad de muerte a 30 días respecto de no aplicarlo?
3. **Pregunta de heterogeneidad**: ¿existen subgrupos (por gravedad, diagnóstico o edad) en los que el efecto del RHC sea neutro o favorable?

### Mapa conceptual del problema

```
Gravedad al ingreso (APACHE, fisiología, diagnóstico) ──► Decisión médica de aplicar RHC ──► Muerte a 30 días
          │                                                                                        ▲
          └────────────────────────────────────────────────────────────────────────────────────────┘
```

La gravedad del paciente influye tanto en la decisión de cateterizar como en el riesgo de muerte. Por eso la comparación directa entre tratados y no tratados confunde el efecto del procedimiento con el efecto de la gravedad (**confusión por indicación**).

### Selección fundamentada del dataset

Se evaluaron siete candidatos: Lending Club, UCI Credit Card Default, Properati, Bank Marketing, Diabetes 130-US Hospitals, Smart Meters London y RHC. Se priorizaron los siguientes criterios, derivados de la ponderación de la rúbrica:

| Criterio | Motivo |
|---|---|
| Tratamiento binario nativo | La estimación por IPW de DoWhy requiere T ∈ {0, 1} |
| Resultado binario | El torneo de la Fase 3 (Naive Bayes, ROC, calibración) es de clasificación |
| Confusión fuerte y documentada | La Fase 4 vale 25 de los 100 puntos |
| Bibliografía de referencia | Permite contrastar el ATE estimado con resultados publicados |
| Bajo costo de ingeniería de datos | El plazo es de una semana |

RHC fue el único candidato que cumplió los cinco criterios. Properati se descartó porque exige construir el tratamiento y los confusores con datos geoespaciales externos, y porque carece de un resultado binario natural. Diabetes 130 queda como alternativa de contingencia.

**Verificación de los requisitos del enunciado:** 5.735 observaciones (≥ 1.500); más de 50 variables explicativas continuas, discretas y categóricas; faltantes no aleatorios (`adld3p`, 75 %), valores centinela (515 pesos registrados como 0 kg), colas pesadas (curtosis 25 en la duración de la internación) y confusión no observable (criterio clínico y centro hospitalario no registrados).

**Riesgo identificado:** el dataset es frecuente en la enseñanza de la inferencia causal. Se consulta a la cátedra antes de avanzar con la Fase 4.

### Hipótesis de trabajo

- **H1 (confusión):** los pacientes tratados presentan mayor gravedad basal que los controles; en consecuencia, la diferencia ingenua de mortalidad sobrestima el daño atribuible al RHC.
- **H2 (efecto causal):** una vez ajustada la confusión observable, el efecto del RHC sobre la mortalidad a 30 días se reduce respecto de la diferencia ingenua. La literatura (Connors et al., 1996; Hirano & Imbens, 2001) sugiere que el efecto ajustado no es favorable.
- **H3 (heterogeneidad):** el efecto del RHC no es homogéneo y depende de la gravedad basal y de la categoría diagnóstica.

---

## Entrada 2. Matriz de decisiones estadísticas y supuestos

**Fecha:** 12/11/2026 · **Firma:** Valentina Bolívar (Backend & ML / Causal Engineer) · **Validación QA:** Emilia Bedners (QA & Validation Engineer) · 12/11/2026

### Fase 1. Auditoría de datos e inferencia clásica

| # | Decisión | Diagnóstico que la motivó | Remediación y justificación |
|---|---|---|---|
| 1 | Recodificar 896 valores centinela como faltantes | Peso de 0 kg (515), frecuencia cardíaca 0 (159), frecuencia respiratoria 0 (136), presión media 0 (80), leucocitos 0 (4), albúmina > 10 g/dL (2) | Son valores físicamente imposibles en un paciente vivo al ingreso. Los ceros del peso subestimaban la media en 9 % e inflaban el desvío en 40 %. Las reglas son declarativas (`src/rhc/cleaning.py`) y están cubiertas por tests. |
| 2 | Conservar un **indicador de faltante** para cada variable recodificada | Los pacientes con ceros en frecuencia cardíaca, respiratoria o presión tienen un APACHE de 78 a 87 frente a 54 del resto | El cero podría ser una medición real en paro cardiorrespiratorio y no un faltante. El indicador preserva esa información sin imponer ninguna de las dos lecturas. |
| 3 | Conservar los ceros de `scoma1`, `urin1` y `edu` | 0 es un valor válido de la escala (sin coma, anuria, sin escolaridad formal) | Distinguir centinelas de ceros legítimos evita destruir información clínica. |
| 4 | Dejar como faltante la duración de internación del único paciente sin fecha de alta | Imputar con la fecha de muerte daba 394 días, con último contacto a los 196 | El registro es inconsistente: imputarlo habría creado el valor máximo del dataset a partir de un error. |
| 5 | Rechazar MCAR para todas las variables con faltantes | Logit $R \sim Z + T$ con test de razón de verosimilitud: $p < 0{,}001$ en los seis casos | La eliminación por filas incompletas sesgaría la muestra. |
| 6 | Excluir `adld3p` (índice ADL, 75 % faltante) | Su ausencia multiplica por 4,9 las chances de muerte a 30 días aun condicionando en $Z$ y $T$ | Evidencia de MNAR: imputar supondría MAR, una hipótesis insostenible. |
| 7 | Excluir `urin1` (diuresis, 53 % faltante) | MAR (depende de $Z$ y $T$, no de $Y$), pero con más de la mitad de los valores ausentes | Una variable imputada en su mayoría aporta más ruido que señal. |
| 8 | Imputar por **mediana estimada solo en train** | Faltantes MAR o MNAR posibles, por debajo del 10 % por variable | Transparente y robusta a colas. Se ajusta dentro del pipeline para que el conjunto de test no filtre información al entrenamiento. |
| 9 | `cat2` faltante = categoría "Sin diagnóstico secundario" | Faltante estructural (79 %) | La ausencia tiene significado clínico. |
| 10 | Fijar $\alpha = 0{,}05$ a priori y calcular la potencia | Con $n_1 = 2.184$ y $n_0 = 3.551$ se detecta $h \geq 0{,}076$ con 80 % de potencia | Con muestras grandes el riesgo no es la falta de potencia sino el sesgo: todo efecto, incluso uno espurio, resulta significativo. Por eso se reportan siempre IC y tamaño de efecto. |
| 11 | No decidir la normalidad con Shapiro-Wilk | Con $n$ entre 270 y 2.490 por grupo, rechaza incluso con asimetría 0,0 y curtosis −0,1 | Se evalúa con asimetría, curtosis y gráficos Q-Q. El TCL respalda la inferencia sobre medias. |
| 12 | Transformar los días de internación con logaritmo | Asimetría 3,9 y curtosis 25 en escala original; 0,2 y −0,2 tras el log | Estabiliza la varianza y acerca los residuos a la normal. Los resultados se reportan como medias geométricas. |
| 13 | Usar t de Welch en lugar de t de Student (H1) | Varianzas distintas entre tratados y controles | Welch no pierde potencia cuando las varianzas son iguales y protege el error de tipo I cuando no lo son. |
| 14 | Usar U de Mann-Whitney para los días de internación (H3) | Curtosis 25: la media es inestable | El test por rangos compara distribuciones sin depender de las colas. |
| 15 | Agrupar cirrosis y los dos cánceres en "Otros" para el ANOVA | Cáncer de colon con $n = 7$ y de pulmón con $n = 39$ | Grupos tan chicos inflan la varianza y vuelven inestables las comparaciones *post-hoc*. |
| 16 | Elegir **ANOVA de Welch** como contraste principal y **Kruskal-Wallis** como respaldo | Levene/Brown-Forsythe rechaza la homocedasticidad ($p = 4{,}4 \times 10^{-16}$) | Welch contrasta medias (en log, medias geométricas), que es el parámetro de interés para planificar camas. Kruskal-Wallis contrasta dominancia estocástica sin supuestos distribucionales. Ambos rechazan $H_0$ y coinciden. |
| 17 | Contrastar Tukey HSD con Bonferroni sobre t de Welch | Tukey supone varianzas iguales, que Levene rechazó | Las 21 decisiones coinciden (14 pares significativos): la conclusión no depende del supuesto violado. |

### Fase 2. Modelo lineal, supuestos de Gauss-Markov y PCA

**Fecha:** 12/11/2026 · **Firma:** Valentina Bolívar (Backend & ML / Causal Engineer) · **Validación QA:** Emilia Bedners (QA & Validation Engineer) · 12/11/2026

| # | Decisión | Diagnóstico que la motivó | Remediación y justificación |
|---|---|---|---|
| 18 | Definir una única partición 80/20, estratificada por mortalidad × tratamiento, con semilla 42 | Las Fases 3 y 4 necesitan el mismo conjunto de test y la misma proporción de tratados | Todo parámetro aprendido (medianas, escalado, rotación del PCA, coeficientes) se estima solo en train. Un test automático verifica que la partición sea disjunta, reproducible y balanceada. |
| 19 | Modelar log(días de internación) por MCO con 68 regresores | Variable continua relevante para planificar camas; la escala logarítmica se justificó en la Fase 1 | Los coeficientes se interpretan como cambios porcentuales: 100·(exp(β) − 1). |
| 20 | Fijar categorías de referencia explícitas en las dummies | El codificador elegía la primera categoría en orden alfabético (por ejemplo, "Coma" como referencia del diagnóstico) | Se usa la categoría más frecuente o basal: insuficiencia respiratoria aguda, sin cáncer, cobertura privada, ingreso más bajo. Los coeficientes quedan interpretables. |
| 21 | Eliminar `malighx` | VIF de 26 en el diseño inicial; coincide con `ca` en el 99,3 % de los pacientes | Es la misma información codificada dos veces. Se conserva `ca`, que distingue el cáncer metastásico. El VIF de las dummies de cáncer cae a menos de 2. |
| 22 | Retener la probabilidad de sobrevida a 2 meses pese a un VIF de 5,2 | Colinealidad estructural: es la salida del modelo pronóstico SUPPORT | Excluirla apenas cambia el coeficiente y el error estándar del APACHE y no mejora el ajuste. La regularización Ridge de la Fase 3 aborda esta colinealidad. |
| 23 | Generar indicadores de faltante solo para peso, frecuencia cardíaca, frecuencia respiratoria y presión media | Con un indicador para cada variable imputada, el de albúmina tenía un único caso: apalancamiento igual a 1 y matriz de diseño de rango deficiente | Un indicador con 1 a 4 casos no aporta información y degenera el modelo. Se conservan los de las cuatro variables cuya pérdida se analizó en la Fase 1. La distancia de Cook máxima bajó de 0,11 a 0,019. |
| 24 | Usar errores estándar robustos HC3 en toda la inferencia del MCO | Breusch-Pagan rechaza la homocedasticidad ($p = 5{,}6 \times 10^{-11}$) | Bajo heterocedasticidad, MCO es insesgado pero sus errores clásicos son inconsistentes. HC3 cambia los errores un 3 % en la mediana y ninguna decisión de significancia. |
| 25 | Usar Breusch-Pagan en lugar de White | Con 68 regresores, la regresión auxiliar de White tendría unos 2.400 términos | El contraste se quedaría sin grados de libertad efectivos. |
| 26 | No rechazar el modelo por el resultado de Jarque-Bera | Rechaza ($p = 0{,}001$) con asimetría 0,13 y curtosis 2,91 | Con $n = 4.587$ el contraste detecta desviaciones irrelevantes; el TCL asegura la normalidad asintótica de los estimadores. |
| 27 | Calcular Durbin-Watson con los residuos ordenados por fecha de admisión | El orden del archivo no tiene significado temporal | Con orden cronológico, DW = 1,98: sin autocorrelación entre pacientes admitidos en fechas cercanas. |
| 28 | Conservar los 9 atípicos con $\lvert r_i \rvert > 3$ | Son internaciones de 150 a 342 días de pacientes que sobrevivieron, con registros coherentes | No son errores de medición. Huber confirma los doce coeficientes clave dentro del IC del MCO. |
| 29 | Ajustar imputación, estandarización y PCA solo sobre train, y aplicar la rotación a ciegas sobre test | Regla estricta del enunciado | La varianza explicada en test con la rotación de train coincide con la de train: la estructura generaliza. |
| 30 | Retener 8 componentes por el criterio de Kaiser | Autovalores mayores a 1; el gráfico de sedimentación no tiene un codo nítido | Explican el 58 % de la varianza. CP1 resume la gravedad y la falla multiorgánica; CP2, el perfil respiratorio crónico frente al cuadro inflamatorio agudo. |

### Fase 3. Torneo predictivo y calibración

**Fecha:** 12/11/2026 · **Firma:** Valentina Bolívar (Backend & ML / Causal Engineer) · **Validación QA:** Emilia Bedners (QA & Validation Engineer) · 12/11/2026

| # | Decisión | Diagnóstico que la motivó | Remediación y justificación |
|---|---|---|---|
| 31 | Encapsular el preprocesamiento dentro de cada modelo (*pipeline*) | En la validación cruzada, imputar o escalar con todo train filtraría información del pliegue de validación | La imputación, el escalado y las dummies se reajustan en cada pliegue. |
| 32 | Seleccionar hiperparámetros por AUC-ROC en `StratifiedKFold` de 5 pliegues | Resultado binario con 33 % de prevalencia | El AUC no depende del umbral, y la estratificación mantiene la prevalencia en cada pliegue (33,4 % a 33,5 %). |
| 33 | Adoptar la penalización L2 (Ridge) frente a L1 (Lasso) | Ambas alcanzan un AUC de 0,76; Lasso anula 12 de 68 coeficientes | Ridge reparte el peso entre variables colineales (APACHE y probabilidad de sobrevida) en lugar de elegir una arbitrariamente. |
| 34 | Reportar odds ratios desde una logística sin penalizar | La penalización sesga los coeficientes hacia cero e invalida los intervalos clásicos | Los OR con IC 95 % se estiman por máxima verosimilitud; el modelo penalizado se usa para predecir. |
| 35 | Implementar un Naive Bayes mixto (gaussiano para continuas y Bernoulli para binarias) | Dataset con 21 variables continuas y 47 binarias | Evita tratar las binarias como gaussianas. El prior de clase se cuenta una sola vez, y un test automático lo verifica. |
| 36 | Elegir el parámetro de poda α por validación cruzada sobre el camino de costo-complejidad | El árbol sin poda sobreajusta (ver Entrada 3) | α = 0,0013 produce un árbol de 16 hojas con AUC de validación de 0,735. |
| 37 | Buscar los hiperparámetros del Random Forest con `RandomizedSearchCV` (20 combinaciones × 5 pliegues) | El espacio de 5 hiperparámetros tiene 576 combinaciones; con 5 pliegues serían 2.880 ajustes, más de una hora de cómputo | La búsqueda aleatoria explora el espacio de manera eficiente; el error out-of-bag converge al de la validación cruzada (0,256 en ambos). |
| 38 | Elegir la logística como modelo final pese a un AUC de validación cruzada levemente menor que el del Random Forest | AUC de validación cruzada de 0,760 frente a 0,773: ventaja de 0,013 | Ante una diferencia tan pequeña se prefiere el modelo interpretable, estable y liviano (13 KB frente a 9 MB). La decisión se toma con train; el bootstrap en test confirma después que la diferencia no es significativa (IC 95 % [−0,009; 0,015]). |
| 39 | Fijar el umbral de decisión por el índice de Youden sobre predicciones fuera de pliegue de train | Con una prevalencia del 33 %, el umbral de 0,5 es conservador y deja sin detectar a muchos pacientes que mueren | Umbral 0,32, elegido sin usar test; medido después en test, el recall sube de 0,42 a 0,67. |
| 40 | Medir la importancia de las variables por permutación en test y no por MDI | MDI ubica una variable de ruido puro en el puesto 9 de 69 (ver Entrada 3) | La permutación mide la pérdida de capacidad predictiva fuera de muestra y relega el ruido al último puesto. |


### Fase 4. Inferencia causal

**Fecha:** 12/11/2026 · **Firma:** Valentina Bolívar (Backend & ML / Causal Engineer) · **Validación QA:**    Emilia Bedners (QA & Validation Engineer) · 12/11/2026

| # | Decisión | Diagnóstico que la motivó | Remediación y justificación |
|---|---|---|---|
| 41 | Usar la muestra completa (5.734 pacientes) para el análisis causal | La partición train/test protege la evaluación predictiva; en la estimación causal no hay predicción fuera de muestra que evaluar | Se maximiza la precisión del ATE. Solo se excluye el paciente sin duración de internación, necesario para ilustrar el sesgo de colisionador. |
| 42 | Expresar todos los efectos como diferencias de riesgo | Resultado binario | Un ATE de 0,05 equivale a 50 muertes adicionales cada 1.000 pacientes: es la escala que entiende la dirección médica. |
| 43 | Identificar con `method_name="maximal-adjustment"` en DoWhy | La búsqueda por defecto de conjuntos backdoor tardaba 70 segundos con 67 confusores | El ajuste máximo coincide con Z, excluye el colisionador y el mediador, y se resuelve en menos de un segundo. |
| 44 | Usar la versión normalizada del IPW (estimador de Hájek) | Las propensiones extremas generan pesos inestables en el estimador de Horvitz-Thompson | Recortar las propensiones fuera de [0,05; 0,95] (6 % de los pacientes) cambia el ATE en menos de medio punto porcentual. |
| 45 | Calcular el intervalo del IPW por bootstrap de 300 réplicas, reestimando la propensión en cada una | DoWhy no reporta un intervalo analítico para la ponderación | El intervalo incorpora la incertidumbre de la estimación de la propensión. |
| 46 | Verificar el balance con diferencias de medias estandarizadas | La validez del IPW depende de que la ponderación balancee Z | Antes de ponderar, 34 covariables superan 0,1 (máximo 0,50); después, ninguna (máximo 0,07). |
| 47 | Complementar las tres refutaciones de DoWhy con un E-value | Las refutaciones no detectan confusores no observados | E-value de 1,67 (1,38 para el límite inferior): se reporta la magnitud de confusión latente necesaria para anular el efecto. |
| 48 | Usar `LinearDML` con Random Forest como modelos de ruido y verificar con `CausalForestDML` | LinearDML produce intervalos válidos para el CATE; el bosque causal no impone linealidad | Ambos métodos coinciden en el ATE y en los patrones por gravedad y diagnóstico. |
| 49 | Reportar la heterogeneidad por gravedad como indicio y no como hallazgo | El intervalo del coeficiente del APACHE en el CATE incluye el cero | El único efecto de subgrupo con intervalo que excluye el cero es la sepsis (+9 puntos porcentuales). |
| 50 | Demostrar las preguntas trampa con una simulación de efecto conocido | El mediador del DAG no se observa en el dataset | La simulación replica la estructura del DAG: el ajuste por Z recupera el efecto verdadero, el ajuste por M deja solo el efecto directo y el ajuste por C sesga la estimación. |

---

## Entrada 3. Autopsia de experimentos fallidos

**Fecha:** 12/11/2026 · **Firma:** Valentina Bolívar (Backend & ML / Causal Engineer) · **Validación QA:** Emilia Bedners (QA & Validation Engineer) · 12/11/2026

### Experimento 1. Árbol de decisión sin restricciones (Fase 3)

- **Hipótesis inicial:** un árbol CART crecido hasta hojas puras capturaría las interacciones entre gravedad, diagnóstico y tratamiento.
- **Resultado:** 707 hojas y 25 niveles de profundidad. Exactitud y AUC de 1,00 en train frente a 0,64 y 0,59 en test, apenas por encima del azar.
- **Diagnóstico:** varianza máxima. Muchas hojas contienen un único paciente, y cada corte profundo ajusta ruido de la muestra de entrenamiento.
- **Corrección:** poda por costo-complejidad con α elegido por validación cruzada. El árbol podado tiene 16 hojas y un AUC en test de 0,744.
- **Aprendizaje:** la exactitud en train de un modelo flexible no informa sobre su capacidad de generalización. La comparación train frente a validación debe hacerse antes de reportar cualquier métrica.

### Experimento 2. Indicadores de faltante para todas las variables imputadas (Fase 2)

- **Hipótesis inicial:** agregar un indicador de faltante para cada variable imputada preservaría la información de la pérdida sin costo.
- **Resultado:** el indicador de albúmina tenía un único caso en train. Ese paciente quedó con apalancamiento $h_{ii} = 1$ (el modelo lo ajustaba de forma exacta), statsmodels advirtió una matriz de diseño de rango deficiente y la distancia de Cook máxima llegó a 0,11.
- **Diagnóstico:** una columna con 1 a 4 casos no aporta información estadística y degenera la estimación.
- **Corrección:** los indicadores se restringieron a las cuatro variables cuya pérdida se analizó en la Fase 1. La distancia de Cook máxima bajó a 0,019 y desapareció la advertencia.
- **Aprendizaje:** el gráfico de influencia (Figura 2.2) detectó un problema de especificación que ningún test de supuestos habría revelado.

### Experimento 3. Naive Bayes como fuente de probabilidades (Fase 3)

- **Hipótesis inicial:** Naive Bayes, rápido y simple, podría ofrecer probabilidades de riesgo comparables a las de la logística.
- **Resultado:** discrimina casi igual (AUC 0,767), pero asigna probabilidades menores a 5 % o mayores a 95 % al 57,5 % de los pacientes de test, y su Brier (0,216) es el peor del torneo.
- **Diagnóstico:** violación del supuesto de independencia condicional. El APACHE, la presión arterial y la probabilidad de sobrevida son funciones parcialmente redundantes de la gravedad, y el modelo multiplica varias veces la misma evidencia.
- **Corrección:** Naive Bayes queda descartado como fuente de probabilidades para la aplicación. Su resultado se documenta como ilustración de las consecuencias del supuesto.
- **Aprendizaje:** un buen AUC no garantiza probabilidades confiables. La calibración debe evaluarse por separado cuando las probabilidades se comunican a un decisor.

### Experimento 4. Importancia de variables por MDI (Fase 3)

- **Hipótesis inicial:** la importancia por impureza (MDI) del Random Forest identificaría las variables que más aportan a la predicción.
- **Resultado:** una variable de ruido gaussiano, agregada como control, ocupó el puesto 9 de 69 en MDI, por encima de todas las variables binarias. La orden de no reanimar, segunda por permutación, quedó en el puesto 18.
- **Diagnóstico:** MDI se calcula sobre los datos de entrenamiento y favorece a las variables con muchos puntos de corte posibles, que en árboles profundos encuentran por azar cortes que reducen la impureza.
- **Corrección:** la importancia se mide por permutación en test, donde el ruido cae al último puesto con importancia nula.
- **Aprendizaje:** incluir una variable de control con importancia conocida (nula) es una forma simple de auditar cualquier medida de importancia.

### Experimento 5. Imputación de la duración de internación con la fecha de muerte (Fase 1)

- **Hipótesis inicial:** para el único paciente sin fecha de alta, usar la fecha de muerte como fin de la internación sería un supuesto razonable.
- **Resultado:** la imputación generaba una internación de 394 días, el máximo del dataset, para un paciente cuyo último contacto registrado fue a los 196 días.
- **Corrección:** el registro se considera inconsistente y su duración queda como faltante.
- **Aprendizaje:** una regla de imputación razonable en general puede fabricar un valor extremo en un caso particular; los extremos deben inspeccionarse uno por uno.

---

## Entrada 4. Formalización del debate causal

**Fecha:** 12/11/2026 · **Firma:** Valentina Bolívar (Backend & ML / Causal Engineer) · **Validación QA:**    Emilia Bedners (QA & Validation Engineer) · 12/11/2026

### Pregunta causal

¿Cuál es el efecto de aplicar cateterismo cardíaco derecho en las primeras 24 horas sobre la probabilidad de muerte a 30 días de un paciente crítico? En notación de Pearl: $P(Y \mid do(T = 1)) - P(Y \mid do(T = 0))$.

### Justificación de cada arco del DAG

| Arco | Justificación | Evidencia en los datos |
|---|---|---|
| $Z_1 \rightarrow T$ | La cobertura médica y el ingreso pueden condicionar el acceso a un procedimiento invasivo y costoso. | Desbalance de cobertura e ingreso entre tratados y controles antes de ponderar. |
| $Z_1 \rightarrow Y$ | La edad y el nivel socioeconómico se asocian con la sobrevida en terapia intensiva. | Edad entre las variables importantes por permutación (Fase 3). |
| $Z_2 \rightarrow T$ | El diagnóstico define la indicación clínica del catéter. | RHC en el 57 % de los pacientes con sepsis frente al 13 % en coma (Fase 1). |
| $Z_2 \rightarrow Y$ | El diagnóstico y las comorbilidades determinan el pronóstico. | Mortalidad a 30 días entre 13 % y más de 50 % según el diagnóstico (Fase 1). |
| $Z_3 \rightarrow T$ | Los pacientes inestables (hipotensos, con mala oxigenación) son los candidatos habituales a monitoreo invasivo. | APACHE 9,8 puntos mayor en los tratados (Fase 1, $d = 0{,}51$). |
| $Z_3 \rightarrow Y$ | La gravedad fisiológica es el principal determinante de la mortalidad. | El APACHE y la probabilidad de sobrevida encabezan todos los rankings de importancia (Fase 3). |
| $Z_4 \rightarrow T$, $Z_4 \rightarrow Y$ | Una orden de no reanimar desalienta procedimientos invasivos y refleja un pronóstico grave. | OR de 2,8 para la mortalidad (Fase 3); desbalance de 0,23 antes de ponderar. |
| $Z_1 \rightarrow Z_2 \rightarrow Z_3 \rightarrow Z_4$ | Las características del paciente preceden al diagnóstico, que precede a la fisiología del día 1 y al pronóstico que se resume a partir de ella. | Colinealidad estructural entre el APACHE, la fisiología y la probabilidad de sobrevida (Fase 2). Estos arcos no alteran el criterio backdoor. |
| $T \rightarrow Y$ | Efecto directo del procedimiento: complicaciones del catéter (arritmias, infecciones, trombosis, lesión vascular). | No separable empíricamente del efecto indirecto en este dataset. |
| $T \rightarrow M \rightarrow Y$ | La lectura del catéter induce cambios de tratamiento (fluidos, inotrópicos, vasopresores) que pueden ser beneficiosos o dañinos. | $M$ no se registra; su papel se demuestra por simulación (Fase 4, sección 4.2). |
| $T \rightarrow C$ | El RHC prolonga la estadía. | +14 % de días de internación con covariables iguales (Fase 2). |
| $Y \rightarrow C$ | La muerte termina la internación. | Correlación de −0,42 entre log(estadía) y muerte a 30 días (Fase 1). |

### Plausibilidad de variables latentes no observadas

- **Criterio del médico tratante ($U$).** Es la amenaza más plausible. El médico decide con información que no queda registrada (aspecto general del paciente, evolución en las horas previas, respuesta a tratamientos iniciales) y que también predice la mortalidad. Si los médicos cateterizan a los pacientes que "ven peor" más allá de lo que miden el APACHE y la fisiología, el efecto dañino estaría sobrestimado. El E-value indica que haría falta un confusor de este tipo con un riesgo relativo de alrededor de 1,7 con el tratamiento y con la muerte para anular el efecto. Es una magnitud plausible pero no trivial, dado que la gravedad medible ya está ajustada.
- **Hospital.** El estudio SUPPORT se realizó en cinco hospitales universitarios con prácticas distintas de uso del catéter. El dataset no identifica el centro. Si un hospital usara más el RHC y además tuviera mayor mortalidad por razones ajenas al procedimiento, sería un confusor no observado.
- **Preferencias del paciente y de la familia.** Están parcialmente capturadas por la orden de no reanimar, pero solo en su forma más extrema.

### Amenazas a la validez interna

1. **Confusión no observada.** Discutida arriba; cuantificada con el E-value.
2. **Momento de la orden de no reanimar.** Se registra el día 1, posiblemente en simultáneo con la decisión de cateterizar. Si el RHC influyera sobre ella, sería un mediador. El análisis de sensibilidad de la sección 4.5 muestra que excluirla del ajuste reduce el ATE de 5,4 a 4,4 puntos porcentuales, que sigue siendo positivo y significativo.
3. **Positividad.** El 6 % de los pacientes tiene propensiones extremas. Recortarlos no modifica el resultado.
4. **Definición temporal del tratamiento.** El tratamiento se define como RHC en las primeras 24 horas. Los pacientes que mueren muy temprano tuvieron menos oportunidad de ser cateterizados y quedan en el grupo control, lo que eleva la mortalidad de los controles. Este sesgo favorecería al RHC: el daño estimado sería, en todo caso, conservador.
5. **Calidad del dato.** Los 896 valores centinela de la Fase 1 se recodificaron como faltantes con un indicador, que preserva la información en caso de que el cero fuera una medición real (paro cardiorrespiratorio).
6. **Validez externa.** Los datos provienen de cinco hospitales universitarios de Estados Unidos entre 1989 y 1994. La práctica del monitoreo hemodinámico cambió desde entonces, y las conclusiones no se trasladan sin más a una terapia intensiva actual. Los ensayos aleatorizados posteriores (Harvey et al., 2005; ESCAPE Investigators, 2005) tampoco encontraron beneficio en mortalidad, lo que refuerza la conclusión principal.

### Conclusión del debate

Bajo los supuestos explicitados, los datos no muestran beneficio del RHC y sí un aumento de la mortalidad a 30 días de entre 4 y 6 puntos porcentuales, robusto a tres estimadores, tres refutaciones y dos especificaciones alternativas del conjunto de ajuste. La principal fragilidad es la posible confusión por el criterio clínico no registrado, cuya magnitud necesaria para anular el efecto es moderada.
