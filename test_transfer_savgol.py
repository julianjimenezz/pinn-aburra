import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
from scipy.signal import savgol_filter

# Añadir directorio raíz al path para importar módulos
current_dir = Path(os.getcwd())
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# Importar módulos del proyecto
from models.gru_pinn import GRUPINN
from models.physics import SoilPhysics

# ==========================================
# 1. CONFIGURACIÓN (EDITAR AQUI)
# ==========================================

# Archivo del modelo entrenado (pesos)
MODEL_PATH = "outputs/models/MedellinOriental_longterm_gru_final.pt"

# Archivo CSV con el NUEVO dataset a probar
CSV_PATH = r"C:/Users/julia/MGEO/THESIS/DEFORMATION_DATA/culstered_landslides_forNN_withRainfall/long_term_prediction/MedellinOriental_10percent.csv"

# Columnas de entrada (deben coincidir con el entrenamiento)
FEATURE_COLS = ['u', 'X', 'Y', 'p_acum_30d', 'p_acum_60d', 'p_acum_90d', 't_days']
TARGET_COL = 'u'

# Parámetros de secuencia
N_STEPS_IN = 5  # Ventana de entrada (5 días)
DT_DAYS = 12.0  # Paso de tiempo aproximado

# Filtro Savitzky-Golay
USE_SAVGOL = True
SAVGOL_WINDOW = 10 
SAVGOL_POLY = 2

# Dispositivo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# ==========================================
# 2. Cargar Datos y Normalizar
# ==========================================
print(f"Leyendo dataset: {CSV_PATH}...")
if not os.path.exists(CSV_PATH):
    print(f"Error: No se encuentra el archivo {CSV_PATH}")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print(f"Filas totales: {len(df)}")

# Verificar si hay múltiples puntos (agrupar por X, Y)
if 'X' in df.columns and 'Y' in df.columns:
    df['point_id'] = df.apply(lambda row: f"{row['X']}_{row['Y']}", axis=1)
else:
    df['point_id'] = "single_point"

unique_points = df['point_id'].unique()
print(f"Puntos únicos encontrados: {len(unique_points)}")

# A.1. FILTRADO SAVITZKY-GOLAY (Opcional)
if USE_SAVGOL:
    print("\nAplicando filtro Savitzky-Golay a la columna 'u'...")
    
    def apply_savgol(group):
        if len(group) > SAVGOL_WINDOW:
            return savgol_filter(group, window_length=SAVGOL_WINDOW, polyorder=SAVGOL_POLY)
        return group
    
    # Ordenar por tiempo antes de filtrar
    df = df.sort_values(by=['point_id', 't_days'])
    
    # Guardar original para comparar
    df['u_raw'] = df['u'].copy()
    
    # Aplicar filtro por grupo
    df['u'] = df.groupby('point_id')['u'].transform(apply_savgol)
    print("✅ Datos suavizados.")

# Normalización
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_features = df[FEATURE_COLS].copy()
scaler.fit(df_features)

u_idx = FEATURE_COLS.index('u')
u_mean = scaler.mean_[u_idx]
u_scale = scaler.scale_[u_idx]
print(f"Estadísticas de 'u': Mean={u_mean:.4f}, Scale={u_scale:.4f}")

def descale_u(u_sc):
    return (u_sc * u_scale) + u_mean

# ==========================================
# 3. Cargar Modelo
# ==========================================
print(f"\nCargando modelo desde: {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: No se encuentra el modelo {MODEL_PATH}")
    sys.exit(1)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model_conf = checkpoint['model_config']
input_size = len(FEATURE_COLS)
hidden_size = model_conf.get('hidden_size', 128)
num_layers = model_conf.get('num_layers', 2)

model = GRUPINN(input_size, hidden_size, num_layers, dropout=0.0).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

soil_physics = SoilPhysics(device=DEVICE)
soil_physics.load_state_dict(checkpoint['soil_physics_state_dict'])

print("✅ Modelos cargados correctamente.")

# ==========================================
# 4. Predicción Autoregresiva
# ==========================================
results = {}

print("\nIniciando predicción autoregresiva con datos suavizados...")
with torch.no_grad():
    for pid in unique_points[:1]: # Probar solo con el PRIMER punto para verificación rápida
        print(f"\nProcesando punto: {pid}")
        
        df_point = df[df['point_id'] == pid].sort_values('t_days').reset_index(drop=True)
        
        if len(df_point) <= N_STEPS_IN:
            print(f"  -> Serie muy corta ({len(df_point)}), saltando.")
            continue
            
        data_scaled = scaler.transform(df_point[FEATURE_COLS])
        
        history_predictions = []
        real_u_values = df_point['u'].values
        
        # Ventana inicial
        current_window = data_scaled[:N_STEPS_IN].copy()
        
        # U previo para física
        last_u_real_scaled = current_window[-1, u_idx]
        last_u_real = descale_u(last_u_real_scaled)
        
        for t in range(N_STEPS_IN, len(df_point)):
            X_in = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Physics
            if len(history_predictions) == 0:
                u_prev_for_physics = last_u_real
            else:
                u_prev_for_physics = descale_u(history_predictions[-1])
            
            u_prev_tensor = torch.tensor([[u_prev_for_physics]], dtype=torch.float32).to(DEVICE)
            u_physics = soil_physics.compute_displacement(u_prev_tensor, dt_days=DT_DAYS)
            u_physics_scaled = (u_physics - u_mean) / u_scale
            
            # NN
            pred_scaled_tensor = model(X_in, u_physics_scaled)
            pred_scaled = pred_scaled_tensor.item()
            
            history_predictions.append(pred_scaled)
            
            # Update window
            if t < len(df_point) - 1:
                new_row = data_scaled[t].copy()
                new_row[u_idx] = pred_scaled # Autoregressive update
                current_window = np.vstack([current_window[1:], new_row])
        
        preds_real_scale = [descale_u(p) for p in history_predictions]
        # Calculate RMSE using smoothed real values
        rmse = np.sqrt(np.mean((real_u_values[N_STEPS_IN:] - np.array(preds_real_scale))**2))
        print(f"  -> Predicción finalizada. RMSE (vs Smoothed): {rmse:.4f}")
        
        results[pid] = {
            "real": real_u_values[N_STEPS_IN:],
            "pred": np.array(preds_real_scale),
            "time": df_point['t_days'].values[N_STEPS_IN:]
        }

print("\nVerificación completada.")
