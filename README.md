# Informe Completo del Proyecto: ARANDU-AI

## Resumen Ejecutivo
**ARANDU-AI** es un modelo de inteligencia artificial enfocado en el análisis temporal de hojas de soja. El repositorio actual contiene la implementación de un **Encoder Fundacional (Base Soy Encoder)** entrenado mediante **Aprendizaje Autosupervisado (Self-Supervised Learning - SSL)**, utilizando una arquitectura basada en **Momentum Contrast (MoCo)**.

El objetivo del proyecto es extraer representaciones vectoriales (embeddings) robustas de imágenes de hojas sin requerir etiquetas manuales, aprovechando grandes volúmenes de datos mediante técnicas avanzadas de contraste. El sistema está diseñado para un alto rendimiento y resiliencia en infraestructuras de múltiples GPUs (como Kaggle).

---

## 1. Arquitectura del Modelo
El modelo se define principalmente en `models/moco.py` y se compone de tres elementos clave:

*   **Backbone (Codificador):** Utiliza una arquitectura **ResNet-50** con pesos iniciales de ImageNet (`IMAGENET1K_V1`), adaptada mediante una capa de identidad en su salida (`fc = nn.Identity()`).
*   **Proyector MLP:** Una red neuronal perceptrón multicapa de 3 capas ocultas (`2048 -> 2048 -> 2048 -> 256`) con *Batch Normalization* y activaciones ReLU, que mapea las características a un espacio de menor dimensión (`dim=256`) donde se calcula el contraste.
*   **Cola de Memoria (MoCoQueue):** Una estructura de datos de tamaño fijo (`K=16384`) que almacena de manera eficiente representaciones de iteraciones pasadas para actuar como "ejemplos negativos" abundantes durante el entrenamiento, separando el tamaño del lote (batch) del número de ejemplos negativos.

---

## 2. Pipeline de Datos y Augmentaciones
El procesamiento de imágenes está fuertemente enfocado en perturbaciones visuales agresivas para que el codificador aprenda invarianzas robustas. Se definen dos vistas de la misma imagen (Query y Key) con las siguientes transformaciones:

*   Recorte y redimensionado aleatorio (`RandomResizedCrop`).
*   Volteo horizontal y **rotaciones personalizadas de 90°** (`RandomRotate90`), muy útiles para imágenes cenitales de hojas.
*   Efectos fotométricos: Variación de color (`ColorJitter`), Conversión a blanco y negro (`RandomGrayscale`) y Solarización aleatoria (`RandomSolarize`).
*   Difuminado de Gauss (`GaussianBlur`).

El dataset de PyTorch personalizado (`MoCoDataset`) incluye un mecanismo de tolerancia a fallos: si una imagen está corrupta o no se puede abrir, el dataloader reintenta cargar otras imágenes aleatorias del conjunto para evitar que el flujo de entrenamiento se detenga de forma abrupta por errores de I/O.

---

## 3. Motor de Entrenamiento (MoCoTrainer)
Ubicado en `engine/trainer.py` y orquestado por el script principal `train.py`, el entrenamiento implementa técnicas computacionales de vanguardia:

*   **Entrenamiento Distribuido (DDP):** Sincronización multi-GPU mediante `DistributedDataParallel`, el uso de `SyncBatchNorm` para regularizar correctamente los lotes divididos y comunicación eficiente entre procesos.
*   **Precisión Mixta (AMP):** Utiliza `autocast` y `GradScaler` para entrenar usando representaciones numéricas a media precisión (FP16/BF16), lo que acelera el cálculo y ahorra memoria VRAM. Oportunamente se fuerzan operaciones matemáticas como la normalización L2 (`F.normalize(z.float())`) al formato de 32-bits para prevenir desbordamientos que lleven a valores inestables (`NaNs`).
*   **Acumulación de Gradientes:** Permite simular lotes (batches) efectivos mucho más grandes acumulando gradientes a través de varios micro-lotes (por defecto 4) antes de actualizar los pesos de la red neuronal.
*   **Optimizador y Planificador de Tasa de Aprendizaje:** Usa descenso de gradiente estocástico (`SGD`) combinado con un planificador que aplica un "calentamiento lineal" (Linear Warmup) seguido de un decaimiento suave en forma de curva coseno (`CosineAnnealingLR`).

---

## 4. Innovaciones Algorítmicas: Auto-Regulación Termodinámica
Un aspecto técnico distintivo del código se encuentra en `engine/scheduler.py`, donde se introdujo un sistema de control dinámico e inteligente para los hiperparámetros del contraste (Temperatura y Momentum):

*   **Calentamiento Térmico (Temp Warmup):** Al arrancar el entrenamiento, el sistema impone una temperatura artificialmente alta (`0.5`). Esto diluye artificialmente el contraste, suavizando y estabilizando los primeros gradientes que recibe la red, reduciendo la temperatura gradualmente a medida que avanza.
*   **Prevención de Colapso (Basada en Uniformidad):** Durante el proceso se monitorea matemáticamente la métrica de **uniformidad** espacial de los embeddings. Si el sistema detecta que los vectores están perdiendo diversidad y empezando a agruparse excesivamente en un mismo lugar (uniformidad muy negativa, `< -4.0`), interviene automáticamente aplicando un *"Temp Boost"* (aumento repentino de temperatura para forzar repulsión entre vectores) y acelera la renovación de la cola de memoria bajando el momentum temporalmente a `0.99`. Esto permite "purgar" la basura de la cola y recuperar la red.

---

## 5. Evaluación Continua y Seguimiento
El modelo incluye módulos analíticos para medir su calidad en vuelo sin interrumpir las épocas:

*   **Evaluación k-NN Inmediata:** Cada bloque de épocas (por defecto 5), el código extrae sin gradientes los vectores de un subconjunto de validación pre-procesado, y ejecuta una clasificación de K-Vecinos más cercanos (`fast_knn`) usando la API rápida de PyTorch (`pdist` o similar) para ofrecer un acercamiento casi en tiempo real de su precisión (`KNN ACC`).
*   **Métricas Internas Explicativas:** El sistema expone constantemente el nivel de **Alineación** (qué tan cerca quedan las vistas idénticas) y **Uniformidad** (qué tan dispersos están todos los datos entre sí en el hiperespacio).
*   **Linear Probing Definitivo:** Al final de la rutina de entrenamiento de MoCo, un protocolo estandarizado (`evaluation/linear_probe.py`) congela por completo los pesos del encoder recién entrenado y acopla un perceptrón lineal clásico, entrenándolo velozmente y exportando el cabezal junto a las métricas definitivas (`Accuracy` y `F1-Score`).

---

## 6. Sistema de Checkpoints Inteligente
El código (`train.py`) se caracteriza por un bloque de manejo de interrupciones que le da una alta resiliencia:
*   Si una sesión se cae (ej. por timeout de GPU o reinicios de máquinas en la nube), al volver a ejecutar `train.py` la función principal escanea automáticamente la existencia del `checkpoint_path`.
*   Si existe, restaura de forma hermética todo el estado: el modelo principal, el modelo llave (Key), los contadores del optimizador, del scheduler, del scaler y, lo más importante, la matriz de la cola de MoCo, continuando la reanudación desde la misma época y bloque de iteración exactos en los que quedó pausado.
*   Incluye un disparador de paro prematuro (*Early Stopping*), el cual se encuentra completamente sincronizado a través de todas las GPUs (`dist.broadcast`) para evitar procesos colgados o *deadlocks*.

---



Para obtener el máximo rendimiento tanto en **eficiencia computacional** (uso de hardware/tiempo) como en **aprendizaje** (calidad de los embeddings) en `config/moco.yaml`. 

Estas recomendaciones están basadas en los estándares de la industria para arquitecturas *Momentum Contrast (MoCo)* entrenadas sobre redes ResNet-50.

### 1. Configuraciones para Mayor Eficiencia (Velocidad y Hardware)
El objetivo aquí es saturar la memoria VRAM de la GPU para paralelizar al máximo y reducir los cuellos de botella del procesador (CPU).

*   **`batch_size`**: Auméntalo de `48` a **`128`** (o `256` si tu GPU tiene 24GB de VRAM). Al usar precisión mixta (`use_amp: True`), una ResNet-50 debería caber cómodamente con batch sizes grandes.
*   **`grad_accum_steps`**: Si aumentas el `batch_size`, reduce esto a **`1`** o **`2`**. Acumular gradientes es útil pero ralentiza el entrenamiento porque requiere múltiples pasadas *forward/backward* por cada paso de optimización.
    *   *Nota:* El tamaño de lote efectivo total (`eff_batch`) de MoCo ideal suele rondar los `256` o `512`. Con `batch_size: 128` y `grad_accum_steps: 2` (por cada GPU) llegas perfectamente a ese valor con mucho mayor velocidad.
*   **`num_workers`**: Súbelo de `4` a **`8`** o incluso **`12`** dependiendo de la cantidad de núcleos de tu CPU en Kaggle/Servidor. Como tus augmentaciones de imagen (Solarize, Rotate, Blur) son agresivas y pesadas para el CPU, esto evitará que la GPU se quede esperando a que lleguen las imágenes.
*   **`use_amp`**: Mantenlo en **`True`** (¡Esencial para el rendimiento!).

### 2. Configuraciones para Mayor Aprendizaje (Calidad de Representación)
En aprendizaje autosupervisado (SSL), más tiempo y más ejemplos negativos casi siempre se traducen en mejores representaciones lineales.

*   **`epochs`**: El aprendizaje contrastivo escala masivamente con el tiempo. `200` está bien para arrancar, pero los modelos de estado del arte entrenan entre **`400`** y **`800`** épocas. Si tienes el tiempo/presupuesto en Kaggle, súbelo mínimo a **`400`**.
*   **`queue` (Ejemplos Negativos)**: El valor actual es `16384`. El paper original de MoCo usa `65536`. Dado que son hojas (un dominio con mucha similitud visual donde los detalles finos importan), subir la cola a **`32768`** o **`65536`** forzará al modelo a comparar contra muchas más variantes y aprender representaciones más discriminativas.
*   **`temp_end` (Temperatura InfoNCE)**: Bajarlo ligeramente de `0.08` a **`0.07`**. El valor `0.07` es el "número mágico" estándar en la mayoría de papers de SSL (MoCo, SimCLR) porque escala los logits de la entropía cruzada a un punto óptimo de dureza (hardness) para los ejemplos negativos.
*   **`momentum_base`**: Si el dataset no es tan gigantesco como ImageNet (1.2M), a veces el encoder *Key* se actualiza demasiado lento con `0.996`. Probar **`0.99`** puede ayudar a que la cola de ejemplos negativos se mantenga más "fresca" con respecto a los pesos actuales del *Query*.

---

### `moco.yaml`:

```yaml
training:
  seed: 42
  epochs: 400              # ⬆️ Mayor aprendizaje (idealmente 800 si hay tiempo)
  warmup_epochs: 10        # ⬆️ Darle más margen a épocas largas
  batch_size: 128          # ⬆️ Mayor eficiencia (saturar VRAM)
  grad_accum_steps: 2      # ⬇️ Reducido para acelerar los steps
  num_workers: 8           # ⬆️ Evitar cuello de botella en CPU
  lr_base: 0.03            # Mantener (es el estándar para batch ~256)
  weight_decay: 0.0001     # Mantener
  use_amp: True            # Mantener
  save_every: 20           # ⬆️ Ajustado a las 400 épocas
  early_stopping_patience: 25 # ⬆️ Darle más paciencia en entrenamientos largos

moco:
  dim: 256
  queue: 65536             # ⬆️ Más negativos duros (o 32768)
  momentum_base: 0.99      # ⬇️ Cola un poco más fresca
  temp_start: 0.15
  temp_end: 0.07           # ⬇️ Estándar de SimCLR/MoCo
  temp_warmup_steps: 150   # ⬆️ Escalar junto con el aumento de la cola
```