import pandas as pd
import os
import numpy as np

ruta_completa = r'C:\Users\lego8\OneDrive\Escritorio\Datathon 2026\dataset_transacciones\hey_productos.csv'
df = pd.read_csv(ruta_completa)

df.drop('es_dato_sintetico', axis=1, inplace=True)

df['fecha_apertura'] = pd.to_datetime(df['fecha_apertura'])
df['fecha_ultimo_movimiento'] = pd.to_datetime(df['fecha_ultimo_movimiento'])

numericas = [
    'limite_credito','saldo_actual','utilizacion_pct',
    'tasa_interes_anual','plazo_meses','monto_mensualidad'
]
df[numericas] = df[numericas].apply(pd.to_numeric, errors='coerce')

df['limite_credito'] = df['limite_credito'].fillna(0)

df['utilizacion_pct'] = df['utilizacion_pct'].fillna(
    df['saldo_actual'] / df['limite_credito'].replace(0, np.nan)
)

df['estatus'] = df['estatus'].fillna('desconocido')

df = df[df['utilizacion_pct'] <= 1.5]

hoy = pd.Timestamp.today()

df['antiguedad_dias'] = (hoy - df['fecha_apertura']).dt.days
df['dias_sin_movimiento'] = (hoy - df['fecha_ultimo_movimiento']).dt.days

df['ratio_deuda'] = df['saldo_actual'] / df['limite_credito'].replace(0, np.nan)

df['es_inactivo'] = (df['dias_sin_movimiento'] > 90).astype(int)

df['carga_mensual_ratio'] = df['monto_mensualidad'] / df['saldo_actual'].replace(0, np.nan)

agg = df.groupby('user_id').agg({
    'producto_id': 'count',
    'saldo_actual': 'sum',
    'limite_credito': 'sum',
    'ratio_deuda': 'mean',
    'utilizacion_pct': 'mean',
    'es_inactivo': 'sum',
    'antiguedad_dias': 'mean',
    'dias_sin_movimiento': 'mean'
}).reset_index()

print(agg.shape)      # filas, columnas
print(agg.columns)    # nombres
print(agg.head())     # primeras filas

#Guardar Archivo
directorio = os.path.dirname(ruta_completa)
ruta_guardado = os.path.join(directorio, 'hey_productos_limpiado.csv')
df.to_csv(ruta_guardado, index=False)


