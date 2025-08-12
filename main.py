# app_v2.py
# Versión corregida del script de Streamlit.

import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ====================================================================================
# 0. Carga de modelo (simulado)
# ====================================================================================

# Carga un modelo de machine learning pre-entrenado
@st.cache_resource
def load_model():
    """Simula la carga de un modelo de ML. Lo entrena con datos dummy."""
    st.info("Cargando y entrenando modelo de Machine Learning...")
    # Creamos un dataset dummy para entrenar el modelo de ejemplo
    X = np.random.rand(1000, 4) * 100
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 150).astype(int)  # Simulación de un resultado de riesgo
    model = LogisticRegression()
    model.fit(X, y)
    st.success("Modelo de Machine Learning de riesgo cargado y listo.")
    return model

# ====================================================================================
# Configuración inicial de la aplicación
# ====================================================================================

st.set_page_config(
    page_title="Pipeline de Datos para Riesgo de Crédito (v2)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛡️ Pipeline de Datos para Riesgo de Crédito (versión avanzada)")
st.markdown("Este dashboard simula un pipeline de datos end-to-end, desde la ingesta hasta el análisis predictivo y la visualización.")
st.markdown("---")

# ====================================================================================
# 1. Simulación de la Ingesta de Datos
# ====================================================================================

st.header("1. 📥 Ingesta y Validación de Datos")

@st.cache_data
def ingest_data():
    """
    Simula la ingesta de datos de clientes y burós de crédito.
    Usamos datos aleatorios para representar las fuentes.
    """
    np.random.seed(42)
    client_ids = range(1, 500001)
    internal_data = pd.DataFrame({
        'client_id': client_ids,
        'antiguedad_meses': np.random.randint(6, 120, len(client_ids)),
        'saldo_promedio': np.random.randint(1000, 100000, len(client_ids)),
        'num_transacciones_ult_3m': np.random.randint(5, 50, len(client_ids)),
        'mora_actual': np.random.choice([0, 30, 60, 90], len(client_ids), p=[0.85, 0.08, 0.05, 0.02])
    })

    # Datos externos de buró de crédito (simulando inconsistencias)
    external_data = pd.DataFrame({
        'client_id': np.random.choice(client_ids, int(len(client_ids) * 0.95), replace=False),
        'score_buro': np.random.randint(300, 850, int(len(client_ids) * 0.95)),
        'num_creditos_abiertos': np.random.randint(0, 10, int(len(client_ids) * 0.95))
    })

    # Introducimos valores inconsistentes para simular errores
    external_data.loc[external_data.sample(frac=0.01).index, 'score_buro'] = -999 

    return internal_data, external_data

internal_df, external_df = ingest_data()

st.success("Datos internos y externos ingestados. Total de clientes: 500,000")

# ====================================================================================
# 2. Validación de Datos (Calidad de datos y Gobierno)
# ====================================================================================

st.header("2. 🧹 Validación y Calidad de Datos")

@st.cache_data
def validate_data(df):
    """Realiza validaciones de calidad de datos y reporta inconsistencias."""
    
    inconsistent_records = df[df['score_buro'] < 0]
    
    # Reporte de anomalías
    if not inconsistent_records.empty:
        st.error(f"¡Alerta! Se encontraron {len(inconsistent_records)} registros con 'score_buro' inválido (< 0).")
        st.info("Estos registros son tratados como inconsistencias y se omiten del análisis, o se imputan.")
    
    # Imputación de datos o eliminación
    validated_df = df[df['score_buro'] >= 0]
    validated_df = validated_df.fillna(500) # Imputación de nulos para clientes sin buró
    
    st.success(f"Validación completada. Se procesan {len(validated_df):,} registros válidos.")
    
    return validated_df

merged_df = pd.merge(internal_df, external_df, on='client_id', how='left')
validated_df = validate_data(merged_df)

st.markdown("---")

# ====================================================================================
# 3. Transformación y Análisis Predictivo
# ====================================================================================

st.header("3. 🧠 Transformación y Análisis Predictivo")

model = load_model()

# Se eliminó el decorador @st.cache_data, que causaba el error
def run_predictions(df, model):
    """
    Simula la ejecución de un modelo de machine learning para predecir el riesgo.
    """
    # Preparamos los datos para el modelo (usando las mismas columnas del entrenamiento)
    df['feature1'] = df['antiguedad_meses']
    df['feature2'] = df['saldo_promedio']
    df['feature3'] = df['num_transacciones_ult_3m']
    df['feature4'] = df['score_buro']
    
    X_test = df[['feature1', 'feature2', 'feature3', 'feature4']].to_numpy()
    
    # Realizamos la predicción de riesgo
    df['riesgo_predicho'] = model.predict_proba(X_test)[:, 1]
    
    return df

final_df = run_predictions(validated_df, model)

st.success("Análisis predictivo de riesgo completado. Se generó un score de riesgo para cada cliente.")
st.write("Resultados del Análisis (primeras 5 filas):")
st.dataframe(final_df[['client_id', 'riesgo_predicho']].head(), use_container_width=True)

st.markdown("---")

# ====================================================================================
# 4. Visualización y Toma de Decisiones
# ====================================================================================

st.header("4. 📊 Visualización y Gobernanza de Datos")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Gobernanza y Roles")
    st.info("""
    **Roles Clave:**
    * **Dueño del Dato (Riesgo):** Define reglas y validaciones.
    * **Ingeniero de Datos (IT):** Diseña y mantiene el pipeline.
    * **Analista de Riesgo:** Consume los datos y toma decisiones.
    * **Auditoría:** Revisa y valida la consistencia del pipeline.
    """)
    st.info("El gobierno asegura la calidad de los datos desde la ingesta hasta el consumo.")

with col2:
    st.subheader("Aprobación de Créditos (Simulación)")
    umbral_riesgo = st.slider("Ajuste del umbral para aprobación de crédito:", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    aprobados = final_df[final_df['riesgo_predicho'] < umbral_riesgo]
    rechazados = final_df[final_df['riesgo_predicho'] >= umbral_riesgo]

    st.metric(label="Créditos Aprobados", value=f"{len(aprobados):,}")
    st.metric(label="Créditos Rechazados", value=f"{len(rechazados):,}")

st.markdown("---")
st.info("Este código simula de manera efectiva un pipeline de datos robusto con validación, análisis predictivo y una clara estructura de gobernanza, aspectos clave de la prueba técnica.")
