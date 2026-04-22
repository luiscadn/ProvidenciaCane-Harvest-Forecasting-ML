import pandas as pd
from sklearn.preprocessing import LabelEncoder


def imputar_por_regla_negocio(df, cols_cero=None, cols_no_aplica=None):
    """
    Imputa columnas específicas según reglas de negocio:
    - cols_cero: columnas numéricas que deben imputarse con 0
    - cols_no_aplica: columnas categóricas que deben imputarse con 'No aplica'
    """
    df_result = df.copy()
    
    if cols_cero is None:
        cols_cero = []
    if cols_no_aplica is None:
        cols_no_aplica = []
    
    for col in cols_cero:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna(0)
    
    for col in cols_no_aplica:
        if col in df_result.columns:
            df_result[col] = df_result[col].fillna('No aplica')
    
    return df_result




def imputar_datos(df):
    """
    Imputa datos nulos a un Dataframe. 
    Lógica de negocio:
    - 'dosismad' se imputa a 0 (No aplicado).
    - Variables numéricas se imputan con la Mediana.
    - Variables categóricas se imputan con la Moda.
    """
    if 'dosismad' in df.columns:
        df['dosismad'] = df['dosismad'].fillna(0)
        
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].fillna(df[col].mode()[0])
            
    return df

def create_target_classes(df, target_col='TCH'):
    """
    Crea la variable objetivo 'Clase_Objetivo' con 3 categorías (Bajo, Medio, Alto)
    balanceadas mediante cuantiles (0.33, 0.66) utilizando pd.qcut.
    """
    df_result = df.copy()
    
    # Creamos las tres clases equilibradas
    df_result['Clase_Objetivo'] = pd.qcut(
        df_result[target_col], 
        q=[0, 0.33, 0.66, 1.0], 
        labels=['Bajo', 'Medio', 'Alto']
    )
    
    # IMPORTANTE: Eliminamos la columna origen para no incurrir en Data Leakage
    columns_to_drop = [target_col]
    # Si tenemos además sacarosa y el target era TCH, a veces también la sacamos:
    if 'sacarosa' in df_result.columns and target_col != 'sacarosa':
        columns_to_drop.append('sacarosa')
        
    df_result = df_result.drop(columns=columns_to_drop, errors='ignore')
    
    return df_result

def apply_label_encoder(df, columns):
    """
    Aplica LabelEncoder a una lista de columnas categóricas en el DataFrame.
    Retorna el DataFrame encodeado y un diccionario con los encoders.
    """
    df_encoded = df.copy()
    encoders = {}
    for col in columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            encoders[col] = le
    return df_encoded, encoders
