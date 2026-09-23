# 🎰 Casino Card Game Recommender & Risk Pattern Detector

Sistema de recomendación por afinidad y detección temprana de patrones de riesgo para casinos en línea especializados en juegos de carta de baraja francesa (Blackjack, Póker Texas Hold'em, Póker 3 Cartas, Baccarat y Pontoon), construido **exclusivamente** a partir de tres señales de interacción no sensibles: `timestamp`, `bet_amount` y `game` — sin datos demográficos ni información personal identificable.

Proyecto desarrollado para el curso **Principios y Tecnologías de Inteligencia Artificial (PTIA)**.

> 📄 El informe técnico completo está en [`main.tex`](main.tex) · 🧪 La implementación y los experimentos están en [`Proyecto_PTIA.ipynb`](Proyecto_PTIA.ipynb)

## Índice

- [Motivación](#motivación)
- [Problema abordado](#problema-abordado)
- [Enfoque técnico](#enfoque-técnico)
- [Stack](#stack)
- [Resultados](#resultados)
- [Consideraciones éticas](#consideraciones-éticas)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Cómo reproducir el experimento](#cómo-reproducir-el-experimento)
- [Limitaciones y trabajo futuro](#limitaciones-y-trabajo-futuro)
- [Autores](#autores)

## Motivación

En un catálogo de casino online limitado a juegos de carta, ofrecer recomendaciones relevantes mejora el descubrimiento de juegos y la experiencia del jugador. El reto de este proyecto es demostrar que esto es posible **con un conjunto de datos deliberadamente mínimo**: solo el momento de la apuesta, el monto apostado y el juego elegido, evitando cualquier señal demográfica o personal.

## Problema abordado

El proyecto explora dos problemas relacionados, y desarrolla en profundidad el primero:

1. **Sistema de recomendación por afinidad** — dado el historial de interacciones `(jugador, juego, timestamp, bet_amount)` de un usuario, predecir qué juegos de carta es más probable que prefiera o pruebe a continuación.
2. **Detector de patrones de riesgo con variables limitadas** *(planteado en el informe como línea de trabajo)* — identificar señales tempranas de comportamiento de riesgo (escaladas de apuesta, frecuencia anómala) usando únicamente las mismas tres variables.

## Enfoque técnico

Pipeline reproducible implementado en el notebook:

1. **Generación de datos sintéticos** — 1,000 usuarios simulados y 25,000 interacciones sobre 5 juegos, con montos de apuesta muestreados de una distribución exponencial por juego.
2. **Ingeniería de señales** — la interacción implícita se pondera como `log1p(bet_amount)` para atenuar outliers de apuestas altas.
3. **Modelo de recomendación — ALS implícito desde cero**: factorización matricial (*Alternating Least Squares*) implementada con NumPy/SciPy sobre la matriz usuario–juego dispersa, sin depender de librerías de recomendación de terceros.
4. **Modelo predictivo — Random Forest**: predice el *siguiente juego* de un usuario a partir de agregados de su historial (`bet_mean`, `bet_sum`, `bet_max`, `bet_std`, `bet_count`), hora modal de juego, último juego jugado y, opcionalmente, los embeddings latentes aprendidos por ALS como features adicionales.
5. **Evaluación con validación temporal** — para cada usuario, su última interacción se reserva como conjunto de prueba y el resto como entrenamiento, simulando la predicción real de "próxima acción".

## Stack

- **Python** (Google Colab / Jupyter)
- `pandas`, `numpy`, `scipy.sparse` — manipulación de datos y matriz usuario–ítem
- `scikit-learn` — `RandomForestClassifier`, `StandardScaler`, `LabelEncoder`, `train_test_split`
- `joblib` — persistencia de artefactos (factores ALS y clasificador)
- LaTeX — informe técnico (`main.tex`)

## Resultados

| Modelo | Métrica | Valor |
|---|---|---|
| ALS (factorización matricial implícita) | Precision@5 | **≈ 0.20** |
| Random Forest (predicción del próximo juego) | Accuracy | **0.28** (vs. 0.20 de línea base aleatoria a 5 clases) |

El modelo ALS captura afinidades latentes razonables dado el tamaño reducido del catálogo (5 juegos). El clasificador de próximo juego supera al azar, confirmando que el monto apostado, su varianza, la hora de juego y el último juego jugado aportan señal útil — aunque el informe deja explícito que capturar dependencias temporales más finas requeriría modelos secuenciales.

## Consideraciones éticas

- **Privacidad by design**: cero variables demográficas o PII; solo `timestamp`, `bet_amount`, `game`.
- **Juego responsable**: el sistema no se optimiza para maximizar gasto; se documentan reglas de negocio que restringirían promociones agresivas ante señales de riesgo.
- **Transparencia**: las decisiones de ponderación y evaluación quedan documentadas para auditoría posterior.
- **Mitigación de sesgos**: se identifica el sesgo de popularidad como riesgo y se proponen estrategias de re-ranking/diversificación como trabajo futuro.

## Estructura del repositorio

```
.
├── Proyecto_PTIA.ipynb   # Implementación: generación de datos, ALS, Random Forest, evaluación
├── main.tex              # Informe técnico completo (LaTeX)
└── README.md
```

## Cómo reproducir el experimento

1. Abrir [`Proyecto_PTIA.ipynb`](Proyecto_PTIA.ipynb) en Google Colab o Jupyter.
2. Instalar dependencias si es necesario: `pip install pandas numpy scipy scikit-learn joblib`.
3. Ejecutar las celdas en orden: generación de dataset → construcción de la matriz usuario–juego → entrenamiento ALS → recomendación → evaluación Precision@K → entrenamiento del Random Forest → guardado de artefactos.

Para compilar el informe en PDF:

```bash
pdflatex main.tex
pdflatex main.tex   # segunda pasada para la tabla de contenidos
```

## Limitaciones y trabajo futuro

- Dataset sintético (no interacciones reales de producción).
- Catálogo reducido a 5 juegos, lo que acota el techo de Precision@K.
- Propuestas de extensión: modelos secuenciales (RNN/Transformers), embeddings más robustos (LightFM, BPR, NCF), mitigación de sesgo de popularidad y ponderación por recencia temporal de las apuestas.

## Autores

- Daniel Julián Peña Bonilla
- Brayan Loaiza Leal
