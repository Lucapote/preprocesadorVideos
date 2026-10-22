---
trigger: always_on
---

# Reglas Operativas del Workflow Local

> **Principio Rector:** *"Tanto algoritmo como sea posible, tanta IA como sea necesaria."*  
> Si un problema puede resolverse con matemáticas, parsing determinista, manipulación de señales o estructuras de datos, **está prohibido recurrir a inferencia estadística o LLMs**[cite: 1].

---

## 1. Jerarquía de Ejecución (Las 3 Capas)

Todo módulo, servicio o decisión técnica debe clasificarse estrictamente en uno de estos tres niveles[cite: 1]:

- **Capa 1: Algoritmo Puro (Determinista y Obligatorio)**
  * **Alcance:** FFmpeg/ffprobe, filtros DSP (loudnorm, ecualización, compresión), expresiones regulares tolerantes, envolventes de onda/RMS, árboles de decisión booleanos, hashes de estado (`sha256`), esquemas Pydantic y serialización JSON[cite: 1].
  * **Criterio:** Costo marginal $0, ejecución local síncrona en microsegundos, reproducible al 100% en pruebas con `pytest`[cite: 1].
- **Capa 2: Modelos Especializados Locales (Inferencia Acotada)**
  * **Alcance:** Modelos entrenados para tareas perceptuales específicas que corren offline en CPU/VRAM: `faster-whisper` (ASR), `Silero VAD` (detección de voz), `sentence-transformers` (embeddings multilingües) o `RNNoise`/`DeepFilterNet` (supresión de ruido)[cite: 1].
  * **Criterio:** El modelo sólo extrae atributos o convierte dominios (ej. audio a texto con timestamps); **no toma decisiones de negocio**[cite: 1]. La salida del modelo vuelve de inmediato a la Capa 1 para ser validada y procesada algorítmicamente[cite: 1].
- **Capa 3: LLM / IA Generativa (Último Recurso por Excepción & Contrato Estricto)**
  * **Alcance:** Invocación de API remota (DeepSeek/OpenAI) reservada exclusivamente para resolver ambigüedades semánticas críticas (ej. tomas huérfanas sin marca con baja similitud de embedding, delimitación de diálogo en ping-pong/trends) o parsear guiones en texto libre desestructurado[cite: 1].
  * **Criterio:** Comunicación bidireccional bajo **JSON estricto** validado por esquemas Pydantic. Prohibido texto libre conversacional tanto en el input como en el output. Si la API falla, excede el timeout (6s) o devuelve un JSON no conforme al schema, el sistema degrada con elegancia a la Capa 1 y envía el caso a `revision/decisiones.json` sin detener el pipeline[cite: 1].

---

## 2. Inmutabilidad y Persistencia de Archivos

- **Brutos de Solo Lectura:** El directorio `brutos/` es estrictamente de lectura. Ningún script modifica, renombra ni sobrescribe los archivos originales del cliente[cite: 1].
- **Caché Regenerable:** Todo archivo dentro de `tomas/` y `storage/temp/` debe considerarse efímero[cite: 1]. Si se borra la carpeta de caché, el sistema debe poder reconstruir el estado ejecutando nuevamente el pipeline desde `brutos/` y `guion.json`[cite: 1].
- **Identificadores Deterministas:** Prohibido usar índices volátiles (`toma_01`, `clip_3`) para el seguimiento de decisiones[cite: 1]. Cada toma se identifica mediante un hash determinista[cite: 1]:
  $$\text{id\_toma} = \text{hash}(\text{archivo\_origen} + \text{timestamp\_inicio\_aprox})$$
[cite: 1]

---

## 3. Integridad Audiovisual

- **CFR Obligatorio en Ingesta:** Antes de cualquier segmentación o corte, el video debe normalizarse a Constant Frame Rate (30.0 fps exactos) y audio a 48 kHz estéreo[cite: 1]. Prohibido recortar sobre material con Variable Frame Rate (VFR) para evitar desincronización A/V[cite: 1].
- **Alineación con I-Frames (Keyframes):** Los cortes exportados en la Fase 5 deben caer en keyframes o forzar reencodificación H.264 rápida en los límites del clip para evitar fotogramas negros o congelados al importar en CapCut[cite: 1].
- **No a los Jump Cuts Automáticos:** La segmentación automática solo elimina el *pre-roll* (desde el aplauso hasta la primera palabra) y el *post-roll* (desde la última palabra hasta la siguiente marca)[cite: 1]. Los silencios y muletillas intermedias no se trocean en el archivo físico final; se reportan como marcas en el `manifest.json`[cite: 1].

---

## 4. Gestión del Audio y Detección de Marcas

- **Búsqueda Acotada:** Prohibido escanear el audio completo en busca de transitorios de aplauso[cite: 1]. El aplauso se busca estrictamente en una ventana temporal relativa a la marca detectada por Whisper ($[-1.5\,\text{s}, +2.5\,\text{s}]$)[cite: 1].
- **Validación Cruzada de Aplauso:** Un aplauso válido requiere simultáneamente:
  1. Un pico de ataque empinado con decaimiento corto en el análisis temporal (Capa 1)[cite: 1].
  2. Confirmación de `Silero VAD` indicando que el instante corresponde a "no-voz" (Capa 2), evitando falsos positivos por sílabas o consonantes oclusivas[cite: 1].
- **Cadena de Audio Secuencial:** El procesamiento de sonido se ejecuta en orden no intercambiable[cite: 1]:
  $$\text{Limpieza (Denoise)} \longrightarrow \text{Ecualización/Compresión} \longrightarrow \text{Normalización EBU R128 (-14 LUFS)}$$
[cite: 1]

---

## 5. Pruebas y Criterios de Transición

- **Desarrollo Guiado por Fases:** No se escribe código de una fase posterior si la anterior no satisface su criterio de aceptación con datos reales[cite: 1].
- **Testabilidad Offline:** Toda función de la Capa 1 y Capa 2 debe contar con tests unitarios en `pytest` que no requieran conexión a internet ni llamadas a APIs de pago[cite: 1].
- **Tolerancia a Fallos en Regex:** El parser de marcas debe procesar texto previamente normalizado (sin tildes, minúsculas, números en palabras convertidos a dígitos arábigos del 1 al 99) antes de evaluar patrones de captura[cite: 1].

---

## 6. Contratos Estrictos con LLM (JSON-In / JSON-Out)

Toda interacción con la Capa 3 debe funcionar como una invocación de procedimiento remoto (RPC) fuertemente tipada:

1. **Entrada Estructurada (JSON Payload):**
   * El LLM **nunca** recibe texto libre descontextualizado.
   * La entrada debe ser un payload JSON validado que incluya únicamente los campos necesarios (ej. lista de tokens indexados `{"index": int, "word": str}`, metadata de la toma y objetivo del bloque).
2. **Salida Estructurada (Strict Structured Output):**
   * El LLM **debe responder exclusivamente con un JSON válido** conforme a un schema Pydantic predefinido (usando `response_format={"type": "json_object"}` o Structured Outputs).
   * Prohibido texto introductorio, explicaciones por fuera del JSON o bloques con etiquetas markdown innecesarias (` ```json `).
3. **Validación Inmediata en Backend:**
   * La respuesta del LLM se valida al instante con `ModeloPydantic.model_validate_json(response_text)`.
   * Si la validación falla (campos faltantes, tipos erróneos o JSON malformado), el evento se captura como excepción, se registra en logs y se activa el fallback determinista de Capa 1.
4. **Prohibición de Flotantes/Timestamps en el LLM:**
   * El contrato JSON prohíbe que el LLM devuelva timestamps en segundos (ej. `12.45s`).
   * El LLM solo devuelve **índices enteros** de los tokens/palabras provistos en la entrada (`first_valid_word_index`, `last_valid_word_index`). La conversión a segundos y el cálculo de padding corresponden exclusivamente a la Capa 1.