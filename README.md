La arquitectura de **ARANDU-AI (SojAI)** se basa en el paradigma **MoCo v3** (*Momentum Contrast version 3*), diseñado específicamente para el aprendizaje de representaciones visuales sin supervisión. Su estructura es una **Red Siamesa Asimétrica**, compuesta por dos ramas de procesamiento paralelas que interactúan para organizar un espacio latente de alta fidelidad.

A continuación, se detalla cada componente y la lógica de ingeniería que sustenta este sistema.


## 1. El Backbone: Extractor de Características ($f_\theta$)

El "corazón" de la red es una **ResNet-50**, seleccionada por su equilibrio entre capacidad expresiva y eficiencia computacional en dispositivos de campo.

* **Modificación Estructural:** Se elimina la capa completamente conectada final (`fc`). En su lugar, se utiliza un *Global Average Pooling* (GAP) que entrega un vector de **2048 dimensiones**.
* **Función:** Transforma los píxeles de la imagen de soja en un tensor latente que contiene información morfológica, cromática y de textura.
* **Dualidad:** Ambas ramas (Query y Key) comparten esta arquitectura base, aunque sus pesos evolucionan de manera distinta.





## 2. El Proyector MLP ($g_\theta$)

Ubicado inmediatamente después del backbone en ambas ramas, el proyector actúa como un "embudo" matemático que traslada el conocimiento general a un subespacio optimizado para el contraste.

* **Arquitectura:** Consta de 3 capas densas (`Linear -> BatchNorm -> ReLU`).
* **Reducción Dimensional:** Mapea las 2048 dimensiones del backbone a un espacio de **256 dimensiones**.
* **Normalización $L_2$:** La salida final se normaliza para que el vector resulte en una magnitud de 1. Esto obliga al modelo a trabajar en la superficie de una **hiperesfera unitaria**, donde la similitud se mide puramente por el ángulo (similitud coseno) y no por la magnitud.





## 3. El Predictor MLP ($q_\theta$): La Clave de la Asimetría

Este componente es exclusivo de la **Red Query (Online)**. Es el bloque que rompe la simetría estructural y funcional del sistema.

* **Propósito:** Su tarea es intentar "predecir" la representación que generará la red Key.
* **Prevención de Colapso:** Al forzar esta asimetría, evitamos que las dos redes se pongan de acuerdo en una solución trivial (como devolver siempre ceros). La red Query debe trabajar activamente para mapear sus características hacia un objetivo móvil pero estable.



## 4. Mecánica de Actualización: Rama Online vs. Momentum

La arquitectura se divide en dos flujos de datos con regímenes de aprendizaje opuestos:

| Característica | Rama Query (Online) | Rama Key (Target/Momentum) |
| :--- | :--- | :--- |
| **Entrada** | Vistas globales y locales | Solo vistas globales |
| **Gradientes** | Sí (Backpropagation activo) | No (Gradientes desactivados) |
| **Actualización** | Optimizador (ej. LARS o AdamW) | Media Móvil Exponencial (EMA) |
| **Componentes** | Backbone + Proyector + Predictor | Backbone + Proyector |

### Actualización por Momentum
Los pesos de la red Key ($\theta_k$) no se calculan mediante el error, sino que "siguen" lentamente a los de la red Query ($\theta_q$) mediante la fórmula:

$$\theta_k \leftarrow m\theta_k + (1 - m)\theta_q$$

Donde **$m$** es el coeficiente de momentum (típicamente **0.999**). Esto garantiza que el diccionario de ejemplos negativos sea coherente y estable en el tiempo, evitando fluctuaciones ruidosas en la pérdida **InfoNCE**.





## 5. El Lazo de Control Geométrico (GeoSat)

Integrado en el flujo de entrenamiento, el módulo `controller.py` supervisa la arquitectura analizando la **matriz de covarianza** de las salidas de 256-d.

* **Monitoreo de Varianza:** Si el controlador detecta que las dimensiones se están "apagando" (colapso dimensional), interviene ajustando la temperatura $\tau$ de la función de pérdida.
* **Lógica PID:** Aplica correcciones proporcionales e integrales para mantener la entropía del espacio latente en niveles óptimos, asegurando que cada una de las 256 dimensiones aporte información útil para el diagnóstico fitopatológico.



## 6. Salida y Evaluación Silenciosa

La arquitectura está diseñada para que, tras el pre-entrenamiento, el backbone se "cristalice". 
1.  **Monitor k-NN:** Evalúa la arquitectura buscando vecinos cercanos en el espacio de 256-d.
2.  **Linear Probe:** Valida si un clasificador lineal simple puede separar las clases (Roya, Sano, Mancha) basándose únicamente en las características de la ResNet-50 congelada.

Esta estructura modular permite que ARANDU-AI extraiga características de una hoja de soja con una fidelidad tal que, al finalizar, el modelo entiende la diferencia entre patógenos biológicamente similares mediante su **huella digital geométrica** en el espacio latente.




---


Para que el núcleo de **ARANDU-AI** (el modelo MoCo v3) sea funcional en un entorno de producción industrial, requiere de una infraestructura periférica robusta. Esta "estructura alrededor de la arquitectura" es lo que transforma un algoritmo matemático en un sistema de ingeniería de software resiliente y escalable.

Podemos dividir esta estructura en cinco pilares fundamentales:


## 1. El Sistema de Ingesta Jerárquica (`utils/`)
En el aprendizaje autosupervisado, el modelo no recibe "datos", recibe "relaciones". La estructura alrededor de la carga de datos está diseñada para forzar la **coherencia estructural**.

* **Orquestación Multi-Crop:** El sistema no carga una imagen, sino que dispara un generador de vistas. Produce simultáneamente vistas globales ($224 \times 224$) para el contexto y vistas locales ($96 \times 96$) para la textura.
* **Pipeline de Invariancia:** La estructura de aumentaciones aplica transformaciones deterministas (como rotaciones del grupo $D_4$) y estocásticas (como el desenfoque gaussiano). Esto asegura que el "ruido" inyectado sea biológicamente plausible para una hoja de soja.




## 2. El Lazo de Control y Resiliencia (`engine/`)
Es el "sistema nervioso" del proyecto. Su función es monitorear, corregir y persistir el estado del entrenamiento.

### GeoSat (Controlador Geométrico)
Esta es la capa de abstracción superior. Actúa como un supervisor de telemetría que:
1.  **Analiza:** Extrae la matriz de covarianza de los embeddings en cada step.
2.  **Calcula:** Evalúa el error de entropía latente.
3.  **Actúa:** Ejecuta un comando PID para ajustar la temperatura ($\tau$) en el módulo de pérdida.



[Image of PID controller block diagram]


### Gestión de Persistencia (Checkpointing)
Implementa una lógica de **Doble Buffer**. El sistema guarda el estado actual (`state_dict`) y mantiene una copia de seguridad de la época anterior. Esto garantiza que, ante un fallo de hardware o de suministro eléctrico, la recuperación ocurra en menos de 60 segundos sin corrupción de archivos.


## 3. La Estructura de Memoria Latente (`MoCoQueue`)
Debido a las limitaciones de VRAM (especialmente al trabajar con hardware como la RTX 4050), el sistema no puede procesar miles de imágenes por batch. La solución estructural es la **Cola de Memoria FIFO**.

* **Desacoplamiento:** Separa el tamaño del diccionario de negativos del tamaño del batch de entrenamiento.
* **Consistencia:** Almacena los vectores generados por la red *Momentum* en pasos anteriores.
* **Actualización:** En cada iteración, los vectores nuevos entran y los más antiguos se descartan, manteniendo el "diccionario" fresco y alineado con la evolución del modelo.


## 4. El Framework de Evaluación No Invasiva (`evaluation/`)
Como el modelo es autosupervisado, la estructura incluye un pipeline de validación paralelo que no interfiere con el entrenamiento, pero proporciona métricas de negocio.

* **k-NN Monitor:** Una estructura de datos de búsqueda rápida (basada en similitud coseno) que clasifica muestras de validación al vuelo para reportar el accuracy de "agrupamiento natural".
* **Linear Probe Gate:** Un módulo que congela el backbone y entrena una capa lineal. Es la prueba de calidad que determina si el modelo está listo para ser "cristalizado" y pasar a la fase de fine-tuning.




## 5. Orquestación Distribuida (DDP)
Para escalar el entrenamiento, la estructura implementa **Distributed Data Parallel (DDP)**. Esto implica una lógica de comunicación entre procesos compleja:

1.  **Sincronización de Gradientes:** Utiliza el algoritmo *All-Reduce* para promediar los pesos entre diferentes procesos.
2.  **Broadcast de Cola:** Asegura que la `MoCoQueue` sea idéntica en todas las réplicas del modelo, evitando que cada instancia aprenda una geometría distinta.
3.  **Manejo de Semillas:** Sincroniza la aleatoriedad de las aumentaciones para mantener la coherencia entre los pares positivos de cada GPU.


### Resumen de la Estructura de Archivos
Esta organización modular es lo que permite la mantenibilidad del proyecto:

| Módulo | Responsabilidad Estructural |
| :--- | :--- |
| `models/` | Definición de la arquitectura siamesa y el proyector. |
| `engine/` | Lógica de entrenamiento, bucles PID y control de fallos. |
| `utils/` | Pipeline de aumentaciones y transformaciones fitopatológicas. |
| `evaluation/` | Monitoreo de k-NN y pruebas de separabilidad lineal. |
| `data/` | Gestión de datasets y loaders distribuidos. |

Esta arquitectura sistémica asegura que el núcleo de IA esté protegido contra anomalías de datos, inestabilidades térmicas del gradiente y fallos de infraestructura, permitiendo un entrenamiento de grado industrial.

