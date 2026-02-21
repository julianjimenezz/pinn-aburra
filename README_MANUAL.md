# Manual Completo del Proyecto PINN (Physics-Informed Neural Network)

Este manual es tu guía definitiva para ejecutar, entender y modificar el proyecto. Aquí se explica paso a paso qué hace cada módulo, en qué orden ejecutarlos y dónde encontrar los resultados.

**Nota:** Este proyecto usa una arquitectura donde la física es parte activa de la predicción, no solo una validación final.

---

## 1. Estructura del Proyecto

Antes de empezar, entiende qué tienes en tus carpetas:

*   **`run_pipeline.py`**: El **Jefe**. Es el script principal que coordina todo (entrenamiento, tuning, análisis). Rara vez necesitas ejecutar otros scripts directamente.
*   **`config.yaml`**: El **Centro de Control**. Aquí defines TODOS los hiperparámetros (capas de la red, parámetros físicos iniciales, rutas de archivos).
*   **`models/`**: La **Fábrica**.
    *   `physics.py`: Contiene las leyes físicas (modelo de Perzyna).
    *   `gru_pinn.py` / `lstm_pinn.py`: Las redes neuronales que aprenden el residuo.
*   **`training/`**: El **Gimnasio**.
    *   `trainer.py`: Donde ocurre el entrenamiento y la validación.
*   **`analysis/`**: El **Laboratorio**.
    *   `group_analysis.py`: Para probar grupos de parámetros predefinidos.
    *   `multi_start.py`: Para probar robustez con inicios aleatorios.
*   **`outputs/`**: Los **Resultados**.
    *   `models/`: Donde se guardan los archivos `.pt` (cerebros entrenados).
    *   `results/`: Archivos CSV con historiales de pérdida y métricas.
    *   `figures/`: Gráficos PNG generados automáticamente.

---

## 2. Flujo de Trabajo: ¿Qué hago primero?

El orden lógico de operación es este:

### Paso 1: Configuración (`config.yaml`)
Abre este archivo y asegúrate de que:
1.  `csv_path`: Apunte a tu archivo de datos actual.
2.  `physics`: Tenga valores razonables para `h` (profundidad) y parámetros iniciales seguros.

### Paso 2: Ejecutar Entrenamiento Básico
Para entrenar un solo modelo y ver si funciona.

```bash
poetry run python run_pipeline.py --mode train
```

**Outputs generados:**
*   `outputs/models/NombreDataset_gru_best_val.pt` (El mejor modelo)
*   `outputs/results/NombreDataset_gru_history.csv` (Excel con la pérdida por época)
*   `outputs/figures/NombreDataset_gru_test_prediction.png` (Gráfico comparativo)

### Paso 3: Análisis de Sensibilidad (Opcional)
Si quieres saber qué parámetros físicos importan más.

```bash
poetry run python run_pipeline.py --mode analyze
```
*Requiere haber completado el Paso 2 primero.*

### Paso 4: Experimentos Avanzados (Group Analysis / Multi-Start)
Si quieres probar hipótesis científicas (ej. "¿Funciona mejor con viscosidad alta o baja?").

---

## 3. Guía Detallada de Scripts y Outputs

### A. El Pipeline Principal (`run_pipeline.py`)

Este script es tu navaja suiza. Usa las "banderas" (flags) para cambiar de modo.

**Ejecución con Poetry:**
Como estás usando un entorno virtual gestionado por poetry, siempre debes anteponer `poetry run` a tus comandos.

**Sintaxis General:**
```bash
poetry run python run_pipeline.py [OPCIONES]
```

**Parámetros Disponibles:**
*   `--mode` o `-m`: El modo de ejecución.
    *   `train`: (Por defecto) Entrena un nuevo modelo.
    *   `tune`: Busca los mejores hiperparámetros (capas, learning rate).
    *   `analyze`: Corre el análisis de sensibilidad sobre un modelo ya entrenado.
*   `--config` o `-c`: Ruta a tu archivo de configuración (por defecto: `config.yaml`).
*   `--csv`: (Opcional) Sobrescribe el dataset definido en el config. Útil para probar otro archivo rápidamente sin editar el YAML.
*   `--model`: (Opcional) Sobrescribe el tipo de modelo (`gru` o `lstm`).

#### Ejemplos de uso comunes:

**1. Entrenamiento Básico (Usando lo que hay en config.yaml)**
```bash
poetry run python run_pipeline.py
```
*(Equivale a `--mode train`)*

**2. Entrenar con un modelo diferente (LSTM en lugar de GRU)**
```bash
poetry run python run_pipeline.py --mode train --model lstm
```

**3. Probar rápidamente otro dataset**
```bash
poetry run python run_pipeline.py --csv "ruta/a/OtroDataset.csv"
```

**4. Buscar Hiperparámetros (Tuning)**
```bash
poetry run python run_pipeline.py --mode tune
```
*(Recuerda: esto solo imprime la mejor configuración en consola, no guarda el modelo)*

**5. Análisis de Sensibilidad**
```bash
poetry run python run_pipeline.py --mode analyze
```
*(Analiza el mejor modelo entrenado del dataset actual)*

**¿Qué pasa si tengo varios modelos (GRU y LSTM) para el mismo dataset?**
Por defecto, tomará el que esté configurado en `config.yaml` (`model_type`).
Si quieres analizar el otro, usa los flags:
```bash
poetry run python run_pipeline.py --mode analyze --model lstm
```

#### Modo Análisis (`--mode analyze`)
Ejecuta un Análisis de Sensibilidad sobre un modelo ya entrenado.

*   **¿Qué modelo carga?**
    Busca automáticamente el archivo `_best_val.pt` en la carpeta `outputs/models`. Es decir, usa la versión del modelo que tuvo el menor error de validación durante el entrenamiento.
    *   Ejemplo: `outputs/models/MedellinOriental_gru_best_val.pt`

*   **Outputs que genera:**
    *   `outputs/results/Nombre_sensitivity.csv`: Tabla que muestra cuánto cambia el error y la física al variar cada parámetro.
    *   `outputs/figures/Nombre_sensitivity.png`: Gráfico de "tornado" o barras que resume visualmente qué parámetro es más crítico.

---

### B. Análisis de Grupos (`analysis/group_analysis.py`)

Úsalo cuando tengas conjuntos de parámetros específicos que quieras comparar (definidos en `config.yaml` bajo `group_analysis`).

**Comando:**
```bash
poetry run python analysis/group_analysis.py --epochs 50 --csv "ruta/al/dataset.csv"
```
*(`--csv` es opcional, si no lo pones usa el de `config.yaml`)*
*(Nota: Aquí ejecutamos el script directamente, es una excepción).*

**Configuración en YAML:**
Asegúrate de tener esto en tu `config.yaml`:
```yaml
group_analysis:
  depths: [10.0, 20.0]  # Opcional: Lista de profundidades a probar
  A:
    eta: 50000.0
    phi: 25.0
  B:
    eta: 1000.0
    phi: 25.0
```

**Outputs:**
*   `outputs/group_analysis/results/NombreDataset_group_analysis_h10.0_FECHA.csv`: Tabla comparativa final.
*   `outputs/group_analysis/models/`: Modelos entrenados (`_best_val.pt` y `_best_physics.pt`) para cada grupo.
*   `outputs/group_analysis/NombreDataset_group_comparison_h10.0_FECHA.png`: Gráfico de barras comparando parámetros iniciales vs. finales.

---

### C. Análisis Multi-Start (`analysis/multi_start.py`)

Úsalo para verificar si el modelo siempre llega a la misma solución física, sin importar dónde empiece. Es crucial para demostrar la "Unicidad" de tu solución en la tesis.

**Comando:**
```bash
poetry run python analysis/multi_start.py --runs 10 --csv "ruta/al/dataset.csv"
```
*(`--csv` es opcional, si no lo pones usa el de `config.yaml`)*

**Configuración en YAML:**
*(Ejecuta 10 entrenamientos independientes desde puntos aleatorios).*

**Outputs:**
*   `outputs/multi_start/results/NombreDataset_multi_start_FECHA.csv`: Tabla con los parámetros finales de las 10 corridas.
*   `outputs/multi_start/NombreDataset_multi_start_distribution_FECHA.png`: Histogramas mostrando la dispersión de los resultados.
    *   *Si las barras son estrechas:* ¡Excelente! Solución robusta.
    *   *Si están muy dispersas:* El modelo no está seguro de los parámetros físicos.

---

## 4. Preguntas Frecuentes sobre Archivos

### ¿Qué hay dentro de un archivo `.pt`?
Ubicación: `outputs/models/`
Son archivos binarios de PyTorch. Contienen:
1.  Los pesos de la red neuronal.
2.  **Los valores aprendidos de los parámetros físicos** (`eta`, `c_prime`, etc.).
3.  El estado del optimizador para poder reanudar el entrenamiento.

### ¿Dónde veo si el modelo aprendió física?
Revisa los archivos CSV en `outputs/results/` o los generados por `group_analysis`. Busca las columnas `final_eta`, `final_phi`, etc., y compáralas con `init_eta`, `init_phi`.
*   Si `final` es diferente de `init`, **el modelo aprendió**.
*   Si son idénticos, algo bloqueó el aprendizaje (posiblemente un learning rate muy bajo o gradientes rotos, pero esto ya lo solucionamos en la versión V2).

### No veo gráficos, ¿qué pasó?
Los gráficos se generan automáticamente al final de `run_pipeline.py`. Si cancelaste el entrenamiento a la mitad (Ctrl+C), no se crearán.
Puedes forzar la creación de gráficos si ya tienes los CSVs usando los scripts en `visualization/`.
