---
trigger: always_on
---

# Reglas Operativas del Workflow Local

> **Principio Rector:** *"Tanto algoritmo como sea posible, tanta IA como sea necesaria."*  
> Si un problema puede resolverse con matemáticas, parsing determinista, manipulación de señales o estructuras de datos, **está prohibido recurrir a inferencia estadística o LLMs**.

---

## 1. Jerarquía de Ejecución (Las 3 Capas)

Todo módulo, servicio o decisión técnica debe clasificarse estrictamente en uno de estos tres niveles:

* **Capa 1: Algoritmo Puro (Determinista y Obligatorio)**
  * **Alcance:** FFmpeg/ffprobe, filtros DSP (loudnorm, ecualización, compresión), expresiones regulares tolerantes, envolventes de onda/RMS, árboles de decisión booleanos, hashes de estado (`sha256`), esquemas Pydantic y serialización JSON.
  * **Criterio:** Costo marginal $0, ejecución local síncrona en microsegundos, reproducible al 100% en pruebas con `pytest`.
* **Capa 2: Modelos Especializados Locales (Inferencia Acotada)**
  * **Alcance:** Modelos entrenados para tareas perceptuales específicas que corren offline en CPU/VRAM: `faster-whisper` (ASR), `Silero VAD` (detección de voz), `sentence-transformers` (embeddings multilingües) o `RNNoise`/`DeepFilterNet` (supresión de ruido).
  * **Criterio:** El modelo sólo extrae atributos o convierte dominios (ej. audio a texto con timestamps); **no toma decisiones de negocio**. La salida del modelo vuelve de inmediato a la Capa 1 para ser validada y procesada algorítmicamente.
* **Capa 3: LLM / IA Generativa (Último Recurso por Excepción)**
  * **Alcance:** Invocación de API remota (DeepSeek/OpenAI) reservada exclusivamente para resolver ambigüedades semánticas críticas (ej. tomas huérfanas sin marca con baja similitud de embedding) o parsear guiones en texto libre desestructurado.
  * **Criterio:** El pipeline debe ser plenamente operativo offline. Si la API falla, se agotan los créditos o se desconecta la red, el sistema degrada con elegancia enviando el caso a `revision/decisiones.json` sin detener el procesamiento.

---

## 2. Inmutabilidad y Persistencia de Archivos

* **Brutos de Solo Lectura:** El directorio `brutos/` es estrictamente de lectura. Ningún script modifica, renombra ni sobrescribe los archivos originales del cliente.
* **Caché Regenerable:** Todo archivo dentro de `tomas/` y `storage/temp/` debe considerarse efímero. Si se borra la carpeta de caché, el sistema debe poder reconstruir el estado ejecutando nuevamente el pipeline desde `brutos/` y `guion.json`.
* **Identificadores Deterministas:** Prohibido usar índices volátiles (`toma_01`, `clip_3`) para el seguimiento de decisiones. Cada toma se identifica mediante un hash determinista:
  $$\text{id\_toma} = \text{hash}(\text{archivo\_origen} + \text{timestamp\_inicio\_aprox})$$

---

## 3. Integridad Audiovisual

* **CFR Obligatorio en Ingesta:** Antes de cualquier segmentación o corte, el video debe normalizarse a Constant Frame Rate (30.0 fps exactos) y audio a 48 kHz estéreo. Prohibido recortar sobre material con Variable Frame Rate (VFR) para evitar desincronización A/V.
* **Alineación con I-Frames (Keyframes):** Los cortes exportados en la Fase 5 deben caer en keyframes o forzar reencodificación H.264 rápida en los límites del clip para evitar fotogramas negros o congelados al importar en CapCut.
* **No a los Jump Cuts Automáticos:** La segmentación automática solo elimina el *pre-roll* (desde el aplauso hasta la primera palabra) y el *post-roll* (desde la última palabra hasta la siguiente marca). Los silencios y muletillas intermedias no se trocean en el archivo físico final; se reportan como marcas en el `manifest.json`.

---

## 4. Gestión del Audio y Detección de Marcas

* **Búsqueda Acotada:** Prohibido escanear el audio completo en busca de transitorios de aplauso. El aplauso se busca estrictamente en una ventana temporal relativa a la marca detectada por Whisper ($[-1.5\,\text{s}, +2.5\,\text{s}]$).
* **Validación Cruzada de Aplauso:** Un aplauso válido requiere simultáneamente:
  1. Un pico de ataque empinado con decaimiento corto en el análisis temporal (Capa 1).
  2. Confirmación de `Silero VAD` indicando que el instante corresponde a "no-voz" (Capa 2), evitando falsos positivos por sílabas o consonantes oclusivas.
* **Cadena de Audio Secuencial:** El procesamiento de sonido se ejecuta en orden no intercambiable:
  $$\text{Limpieza (Denoise)} \longrightarrow \text{Ecualización/Compresión} \longrightarrow \text{Normalización EBU R128 (-14 LUFS)}$$

---

## 5. Pruebas y Criterios de Transición

* **Desarrollo Guiado por Fases:** No se escribe código de una fase posterior si la anterior no satisface su criterio de aceptación con datos reales.
* **Testabilidad Offline:** Toda función de la Capa 1 y Capa 2 debe contar con tests unitarios en `pytest` que no requieran conexión a internet ni llamadas a APIs de pago.
* **Tolerancia a Fallos en Regex:** El parser de marcas debe procesar texto previamente normalizado (sin tildes, minúsculas, números en palabras convertidos a dígitos arábigos del 1 al 99) antes de evaluar patrones de captura.