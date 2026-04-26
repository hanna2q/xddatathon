import numpy as np
import pandas as pd

"""# **DATASETS NO TRANSACCIONES**"""

df_clientes = pd.read_csv('/content/drive/MyDrive/hey_clientes.csv')
df_clientes.head()

df_clientes.describe()

df_clientes.nunique()

df_productos = pd.read_csv('/content/drive/MyDrive/hey_productos.csv')
df_productos.head()

df_productos.shape
df_productos['user_id'].nunique()

df_productos.describe()

df_productos.nunique()

"""# **TRANSACCIONES**

# ***Carga del dataset***
"""

df_transacciones = pd.read_csv('/content/drive/MyDrive/hey_transacciones.csv')
df_transacciones.head()

df_transacciones.head()

df_transacciones['descripcion_libre'].head()

df_transacciones['descripcion_libre'].unique()

df_transacciones.nunique()

"""# ***LIMPIEZA DE DATOS***"""

df_transacciones.isnull().sum()

df_transacciones.shape

"""Considerando los datos se decide eliminar motivo_no_procesado porque solo aporta ruido, lo mismo con descripción_libre que además no consideramos muy importante porque ya iba muy relacionado con la actividad, canal, etc. También meses diferidos porque consideramos que, aunque se puede volver una flag, no aporta mucho de si tienen o no meses diferidos, no aporta mucho. Para las otras columnas con datos nulos se aplicará limpieza o se convertirá en otro dato en si.

"""

df_transacciones.drop(['motivo_no_procesada', 'descripcion_libre', 'meses_diferidos'], axis=1, inplace=True)

"""![Captura de pantalla 2026-04-25 180752.png](data:image/png;base64,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)




"""

df_transacciones['comercio_nombre'] = df_transacciones['comercio_nombre'].fillna('desconocido')

#manejar nulos en ciudad_transacción
df_transacciones['ciudad_transaccion'] = df_transacciones['ciudad_transaccion'].fillna('desconocida')

#manejar nulos en dispositivo
df_transacciones['dispositivo'] = df_transacciones['dispositivo'].fillna('desconocido')

#manejar nulos en cashback
df_transacciones['cashback_generado'] = df_transacciones['cashback_generado'].fillna(0)

#se decidió usar fecha_hora en lugar de hora y dia por separado asi que también se eliminan
df_transacciones = df_transacciones.drop(columns=['hora_del_dia', 'dia_semana'])

"""# **VARIABLES ÚTILES**

- tipo_operacion
* canaL
* monto
* categoria_mcc
* estatus
* motivo_no_procesada
* intento_numerO
* meses_diferidos
* cashback_generado
* es_internacional
"""

df_transacciones['monto'].hist(bins=50)

df_transacciones.groupby('user_id')['monto'].sum().sort_values(ascending=False)

"""Crosstabs para ver frecuencias"""

pd.crosstab(df_transacciones['categoria_mcc'], df_transacciones['estatus'])

"""# **FUNCIONES PARA FEATURES**"""

# 1.PREPARACIÓN
def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Fechas
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], errors='coerce')
    df = df.dropna(subset=['fecha_hora'])

    df['hora'] = df['fecha_hora'].dt.hour
    df['dia_semana'] = df['fecha_hora'].dt.dayofweek
    df['es_fin_semana'] = df['dia_semana'].isin([5, 6]).astype(int)

    # Flags útiles
    df['es_fallida'] = (df['estatus'] != 'aprobada').astype(int)
    df['tiene_cashback'] = (df['cashback_generado'] > 0).astype(int)

    return df

# 2. FEATURES NUMÉRICAS BASE
def build_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby('user_id').agg(
        total_transacciones=('transaccion_id', 'count'),
        monto_total=('monto', 'sum'),
        monto_promedio=('monto', 'mean'),
        monto_std=('monto', 'std'),
        monto_max=('monto', 'max'),
        monto_min=('monto', 'min'),
    )

    return agg

# 3. FEATURES DE COMPORTAMIENTO
def build_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby('user_id').agg(
        tasa_fallos=('es_fallida', 'mean'),
        tasa_cashback=('tiene_cashback', 'mean'), #no creo que sea útil
        tasa_internacional=('es_internacional', 'mean'),
        tasa_atipico=('patron_uso_atipico', 'mean'),
        intento_promedio=('intento_numero', 'mean'),
    )

    return agg

# 4. FEATURES TEMPORALES
def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values(['user_id', 'fecha_hora'])

    # Tiempo entre transacciones
    df_sorted['diff_time'] = df_sorted.groupby('user_id')['fecha_hora'].diff().dt.total_seconds()

    agg = df_sorted.groupby('user_id').agg(
        tiempo_medio_entre_tx=('diff_time', 'mean'),
        actividad_fin_semana=('es_fin_semana', 'mean'),
        actividad_nocturna=('hora', lambda x: ((x >= 22) | (x <= 6)).mean())
    )

    return agg

# 5. FEATURES CATEGÓRICAS (DISTRIBUCIONES)
def build_categorical_features(df: pd.DataFrame, col: str, prefix: str, top_n: int = 5) -> pd.DataFrame:
    top_categories = df[col].value_counts().nlargest(top_n).index

    df_filtered = df[df[col].isin(top_categories)]

    dummies = pd.get_dummies(df_filtered[col], prefix=prefix)

    df_cat = pd.concat([df[['user_id']], dummies], axis=1)

    agg = df_cat.groupby('user_id').mean()

    return agg

# 6. FEATURES DE DIVERSIDAD
def build_diversity_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby('user_id').agg(
        diversidad_comercios=('comercio_nombre', 'nunique'),
        diversidad_categorias=('categoria_mcc', 'nunique'),
        diversidad_ciudades=('ciudad_transaccion', 'nunique'),
    )

    return agg

# 7. PIPELINE PRINCIPAL
def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_transactions(df)

    features = []

    features.append(build_numeric_features(df))
    features.append(build_behavior_features(df))
    features.append(build_temporal_features(df))
    features.append(build_diversity_features(df))

    # Categóricas importantes
    features.append(build_categorical_features(df, 'categoria_mcc', 'mcc'))
    features.append(build_categorical_features(df, 'canal', 'canal'))
    features.append(build_categorical_features(df, 'tipo_operacion', 'tipo'))

    # Merge final
    user_features = pd.concat(features, axis=1)

    # Limpieza final
    user_features = user_features.fillna(0)

    return user_features

#implementación de features
user_features = build_user_features(df_transacciones)
user_features.to_csv('prod_user_features.csv')
user_features.head()

user_features.nunique()