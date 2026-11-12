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

**Fecha:** 12/11/2026 · **Firma:** Valentina Bolívar (Backend & ML / Causal Engineer) · **Validación QA:** Emilia Bedners (validado el 12/11/2026)

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

**Fecha:** 12/11/2026 · **Firma:** Valentina Bolívar (Backend & ML / Causal Engineer) · **Validación QA:** Emilia Bedners (pendiente)

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

