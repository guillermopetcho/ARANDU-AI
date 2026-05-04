Este es un texto técnico, exhaustivo y de nivel profesional diseñado específicamente para servir como el cuerpo principal del archivo `README.md` en un repositorio de GitHub de clase industrial. Está estructurado para que cualquier reclutador o ingeniero senior de IA pueda entender la complejidad y el rigor del proyecto.

---

# ARANDU-AI (SojAI): Sistema de Visión Computacional Autosupervisado para Fitopatología Digital

## 1. Introducción y Propósito del Proyecto
**ARANDU-AI** representa un hito en la aplicación de la Inteligencia Artificial al sector agropecuario. A diferencia de los sistemas tradicionales de clasificación que dependen de miles de etiquetas manuales (propensas al error humano y costosas de producir), ARANDU-AI implementa un paradigma de **Aprendizaje Autosupervisado (SSL)**. El sistema es capaz de "aprender" por sí mismo la morfología, texturas y patrones de las enfermedades de la soja, extrayendo representaciones latentes de alta fidelidad antes de siquiera ver una etiqueta de clase.



El objetivo final es cristalizar un **Encoder Universal de Soja** que pueda ser desplegado en dispositivos de borde (Edge AI) para diagnósticos en tiempo real con una precisión superior al 95% utilizando solo una fracción de los datos etiquetados requeridos por otros modelos.

---

## 2. El Motor Teórico: Momentum Contrast (MoCo v3) e InfoNCE
El sistema se fundamenta en la arquitectura **MoCo v3**, que reformula el aprendizaje de imágenes como una tarea de búsqueda en un diccionario dinámico. El modelo aprende mediante la comparación de pares positivos (versiones aumentadas de la misma hoja) frente a un conjunto masivo de distractores (pares negativos).

### 2.1 La Función de Pérdida InfoNCE
Para organizar el espacio latente, utilizamos la pérdida **InfoNCE (Information Noise-Contrastive Estimation)**, la cual maximiza la información mutua entre representaciones:

$$L_{q, k^+, \{k_i\}} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\sum_{i=0}^{K} \exp(q \cdot k_i / \tau)}$$

Donde $\tau$ es un parámetro de **Temperatura Dinámica** controlado por un scheduler térmico. Esta temperatura regula la concentración de los vectores en la hiperesfera unitaria, permitiendo que el modelo se enfoque en "ejemplos negativos difíciles" (hard negatives) durante las fases finales del entrenamiento.



---

## 3. Arquitectura Estructural: Redes Siamesas Asimétricas
La implementación técnica en `models/moco.py` utiliza una arquitectura de doble rama con asimetría estructural para prevenir el colapso dimensional.



### 3.1 Backbone ResNet-50 Modificada
Utilizamos una **ResNet-50** como extractor base. Hemos eliminado la capa totalmente conectada (`fc`) y la hemos sustituido por una capa de identidad para preservar un vector de características denso de 2048 dimensiones. Este vector captura desde gradientes de color (clorosis) hasta estructuras geométricas complejas (pústulas).



### 3.2 Proyector y Predictor MLP
El sistema mapea las 2048 dimensiones a un espacio de contraste de 256 dimensiones a través de un **Proyector MLP** de 3 capas. La rama **Query (Online)** incluye adicionalmente un **Predictor MLP**, que rompe la simetría y obliga a la red a predecir la representación de la rama **Key (Momentum)**, la cual se actualiza mediante una **Media Móvil Exponencial (EMA)** para garantizar la estabilidad del diccionario.



---

## 4. Ingeniería de Datos y Pipeline Fitopatológico
En SSL, las aumentaciones de datos son el equivalente a las etiquetas. El módulo `utils/` implementa una estrategia de **Multi-Crop** inspirada en DINO:

1.  **Vistas Globales (224x224):** Capturan la arquitectura de la hoja y la disposición general de las lesiones.
2.  **Vistas Locales (96x96):** Se enfocan en texturas microscópicas (esporas, halos necróticos).



El pipeline aplica aumentaciones espaciales (rotaciones de 90° y flips) esenciales para imágenes cenitales, y un **Color Jittering** estrictamente controlado para no destruir la jerarquía biológica del color, vital en el diagnóstico fitopatológico.



---

## 5. Innovación Crítica: Controlador Geométrico (GeoSat)
Una de las contribuciones más audaces de ARANDU-AI es el módulo `engine/controller.py`. Este actúa como un sistema de soporte vital que monitorea la **entropía del espacio latente** en tiempo real.



### 5.1 Lazo de Control PID
Si el sistema detecta que el modelo está sufriendo un colapso dimensional (todos los vectores se parecen), GeoSat activa un bucle de control **PID (Proporcional-Integral-Derivativo)**:

$$u(t) = K_p e(t) + K_i \int e(t) dt + K_d \frac{de(t)}{dt}$$

El controlador ajusta la temperatura $\tau$ o inyecta un "shock térmico" para forzar la repulsión de los vectores, asegurando que el entrenamiento sea resiliente y autónomo. Incorpora además una lógica de **Histéresis (Anti-Flapping)** para ignorar ruidos momentáneos en los datos.



[Image of PID controller block diagram]



[Image of hysteresis loop]


---

## 6. Protocolos de Evaluación y Calidad
Para validar el progreso sin usar etiquetas durante el pre-entrenamiento, el sistema realiza "Evaluaciones Silenciosas":

* **k-NN Monitor:** Busca los vecinos más cercanos en el espacio de 256-d para medir el agrupamiento natural de las enfermedades.
* **Linear Probe:** Congela el Backbone y entrena una única capa lineal. Si el accuracy es alto, se demuestra que las características extraídas son **linealmente separables** y de alta fidelidad semántica.




---

## 7. Conclusión y Futuro
ARANDU-AI no es solo un modelo; es un ecosistema de ingeniería diseñado para la resiliencia en el campo. Gracias al entrenamiento autosupervisado y al control geométrico, el sistema produce embeddings que son robustos ante cambios de iluminación, variedad de sensores y condiciones climáticas, estableciendo un nuevo estándar en la **Fitopatología Digital Inteligente**.