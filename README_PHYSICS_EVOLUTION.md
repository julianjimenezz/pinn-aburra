# Diferencias Clave: Física en la Pérdida vs. Física en la Predicción

Este documento aclara la diferencia fundamental entre tu versión anterior del modelo y la actual, conectando la explicación conceptual con **tus archivos de código actuales**.

## 1. El Enfoque Anterior: "Física solo en la Pérdida" (Regularización)

En tu implementación anterior, la red neuronal (LSTM/GRU) hacía todo el trabajo sola. La física era solo un "castigo" o regularización al final.

### Cómo funcionaba:
1.  **Predicción:** La Red Neuronal recibía los datos (**X**) y predecía el desplazamiento (**u_pred**).
    
    `u_pred = RedNeuronal(X)`
    
    *Nota: La física NO participaba aquí.*

2.  **Cálculo del Error:** Se calculaba el error de datos y se le sumaba un "error físico".
    
    `Loss = ErrorDatos(u_pred, u_real) + lambda * ErrorFisico(u_pred, ParámetrosFijos)`

### Por qué fallaba (El problema de la invarianza):
*   **La Red podía ignorar la física:** Si la red neuronal es lo suficientemente potente (y las GRU/LSTM lo son), puede aprender a predecir **u_real** casi perfectamente minimizando solo el `ErrorDatos`.
*   **Parámetros irrelevantes:** Como la física no generaba la predicción, sino que solo la "juzgaba" al final, el modelo podía encontrar un mínimo local donde ignoraba casi por completo el término físico. Por eso, cambiar **h** o **eta** (viscosidad) no cambiaba la predicción: la red simplemente compensaba el cambio ajustando sus pesos internos para seguir satisfaciendo el `ErrorDatos`.

---

## 2. El Enfoque Actual: "Física en la Predicción" (Aprendizaje Residual)

La diferencia crítica es que ahora el modelo físico es **parte activa** de la predicción. No está solo "mirando" desde la función de pérdida; está **generando** la base de la predicción.

### ¿Dónde ocurre esto en tu código?

Todo sucede en dos archivos principales: `models/physics.py` (el motor físico) y `training/trainer.py` (el director de orquesta).

#### Paso 1: Predicción Física Base (`models/physics.py`)
Primero, el modelo físico calcula cuánto debería moverse el terreno basándose puramente en las leyes de Perzyna y los parámetros actuales (**h**, **eta**, **c_prime**, **phi**).

*   **Archivo:** `models/physics.py`
*   **Función:** `compute_displacement`
*   **Código clave:**
    ```python
    # Calcula la velocidad de deformación según el modelo de Perzyna
    gamma_dot = (1.0 / params['eta']) * torch.pow(F.softplus(overstress), params['n'])
    
    # Calcula el desplazamiento físico (u_physics)
    u_physics = u_prev + velocity_m_per_day * dt_days
    ```

#### Paso 2: Integración en el Bucle de Entrenamiento (`training/trainer.py`)
Aquí es donde ocurre la magia. Antes de preguntarle cualquier cosa a la red neuronal, el entrenador calcula esa predicción física base y se la entrega a la red.

*   **Archivo:** `training/trainer.py`
*   **Función:** `_train_epoch` (y `_compute_u_physics_scaled`)
*   **Código clave:**
    ```python
    # 1. Calcular la línea base física (usando el modelo de physics.py)
    u_physics_scaled = self._compute_u_physics_scaled(X_b)
    
    # 2. Pasar esa base a la red neuronal (GRU/LSTM)
    # La red ahora recibe 'u_physics_scaled' como entrada adicional
    u_pred_scaled = self.model(X_b, u_physics_scaled)
    ```

#### Paso 3: Aprendizaje Residual (`models/gru_pinn.py` o `lstm_pinn.py`)
Dentro de la red neuronal, la predicción física no se ignora. Se suma a la salida de la red. La red neuronal está forzada a predecir solo el **residuo** (la diferencia entre la física y la realidad), no el valor total desde cero.

*   **Archivo:** `models/gru_pinn.py` (o LSTM)
*   **Función:** `forward`
*   **Concepto:** `u_final = u_fisica + delta_u_red`

### Por qué esto arregla todo:
1.  **La Física es la base:** El modelo se ve obligado a usar los parámetros físicos (**h**, **eta**, etc.) para acercarse lo más posible a la realidad (**u_real**).
2.  **Incentivo para aprender:** Si los parámetros físicos son incorrectos (ej. **eta** muy alto), **u_fisica** en el Paso 1 será horrible. Esto obliga a la Red Neuronal en el Paso 3 a trabajar extra para compensar.
3.  **Optimización Eficiente:** El optimizador prefiere ajustar los parámetros físicos (que son pocos y afectan globalmente) para mejorar **u_fisica**, en lugar de forzar a la red neuronal a aprender correcciones complejas para todo.

## Resumen Visual

| | Versión Anterior | Versión Actual |
| :--- | :--- | :--- |
| **Rol de la Física** | Juez (función de pérdida) | Protagonista (genera la predicción base en `physics.py`) |
| **Flujo de Datos** | `Entrada -> Red -> Salida` | `Entrada -> Física + Red -> Salida` (en `trainer.py`) |
| **Si la física falla...** | La red la ignora y memoriza los datos. | La predicción base sale mal y el error se dispara. |
| **Impacto de Parámetros** | Nulo o muy bajo. | **Crítico:** Definen la tendencia principal. |
