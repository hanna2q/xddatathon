import pandas as pd
import numpy as np
import os

# CARGA DE DATOS
base_path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_path, 'hey_clientes.csv')
df_clientes = pd.read_csv(path)

# LIMPIEZA INICIAL DE NOMBRES Y COLUMNAS
df_clientes.columns = df_clientes.columns.str.strip()

columnas_basura = ['idioma_preferred', 'idioma_preferido', 'ciudad']
df_clientes = df_clientes.drop(columns=[c for c in columnas_basura if c in df_clientes.columns], errors='ignore')

# TRANSFORMACIÓN ORDINAL: NIVEL EDUCATIVO
orden_educativo = {
    'Sin estudios': 0, 'Primaria': 1, 'Secundaria': 2, 
    'Preparatoria': 3, 'Universidad': 4, 'Postgrado': 5
}
if 'nivel_educativo' in df_clientes.columns:
    df_clientes['nivel_educativo_num'] = df_clientes['nivel_educativo'].str.strip().map(orden_educativo).fillna(0).astype(int)

# VARIABLES BINARIAS (True/False a 1/0)
cols_booleanas = ['es_hey_pro', 'nomina_domiciliada', 'recibe_remesas', 'usa_hey_shop', 'tiene_seguro', 'patron_uso_atipico']
for col in cols_booleanas:
    if col in df_clientes.columns:
        df_clientes[col] = df_clientes[col].astype(str).str.strip().str.lower().map({
            'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0
        }).fillna(0).astype(int)

# VARIABLES NOMINALES: ONE-HOT ENCODING
columnas_nominales = ['sexo', 'ocupacion', 'estado', 'canal_apertura', 'preferencia_canal']
df_clientes = pd.get_dummies(df_clientes, columns=[c for c in columnas_nominales if c in df_clientes.columns], 
                             prefix=columnas_nominales, dtype=int)

# TRATAMIENTO DE VALORES NUMÉRICOS
cols_num = ['ingreso_mensual_mx', 'score_buro', 'satisfaccion_1_10', 'edad', 'antiguedad_dias']
for col in cols_num:
    if col in df_clientes.columns:
        df_clientes[col] = pd.to_numeric(df_clientes[col], errors='coerce')
        df_clientes[col] = df_clientes[col].fillna(df_clientes[col].median())

# LIMPIEZA FINAL
df_clientes_limpio = df_clientes.drop(columns=['nivel_educativo'], errors='ignore')

# Convertir cualquier residuo booleano a entero
for col in df_clientes_limpio.select_dtypes(include=['bool']).columns:
    df_clientes_limpio[col] = df_clientes_limpio[col].astype(int)

# RESULTADOS
print("¡Limpieza completada con éxito!")
print("-" * 30)
print(f"Total de columnas finales: {df_clientes_limpio.shape[1]}")
print("\nConteo de tipos de datos:")
print(df_clientes_limpio.dtypes.value_counts())

# PREVISUALIZACIÓN SEGURA (Imprime las primeras 5 columnas que encuentre para no fallar)
print("\nVista previa de las primeras columnas:")
print(df_clientes_limpio.iloc[:, :5].head())

# GUARDAR DATASET LIMPIO
ruta_guardado = os.path.join(base_path, 'hey_clientes_limpios.csv')

try:
    df_clientes_limpio.to_csv(ruta_guardado, index=False)
    print(f"\n¡Archivo guardado con éxito en: {ruta_guardado}")
except PermissionError:
    print("\n[!] ERROR: No se pudo guardar. Cierra el archivo 'hey_clientes_limpios.csv' en Excel y vuelve a correr el código.")