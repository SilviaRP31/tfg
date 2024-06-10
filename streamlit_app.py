import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess
import gensim.downloader as api
"""
# Welcome to Calimana!
"""

@st.cache
def load_data():
    df = pd.read_excel("answ.xlsx")
    df['Edad paciente'] = df['Edad paciente'].astype(str)  
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    return df

df = load_data()

# Preprocesar los datos
df_partial = df.sample(frac=0.8)
motivo_paciente_pos = df_partial['Edad paciente'] + ' ' + df_partial['Motivo visita paciente']

# Crear la matriz TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(motivo_paciente_pos)

# Calcular la matriz de similitud de coseno
cosine_similarity_matrix = cosine_similarity(tfidf_matrix)
similarity_df = pd.DataFrame(cosine_similarity_matrix, index=df_partial.index, columns=df_partial.index)

def get_top_3_psychologists(age, reason):
    patient_info = age + ' ' + reason
    tfidf_matrix_input = tfidf.transform([patient_info])
    cosine_similarity_input = cosine_similarity(tfidf_matrix_input, tfidf_matrix)
    similarity_scores = cosine_similarity_input[0]
    top_3_indices = similarity_scores.argsort()[-3:][::-1]
    top_3_psicologos_info = df_partial.iloc[top_3_indices]
    return top_3_psicologos_info[similarity_scores[top_3_indices] > 0]

# Título de la aplicación
st.title('Asignación de Psicólogos a Pacientes')

# Entradas del usuario
edad_paciente = st.text_input('Introduce la edad del paciente')
motivo_visita_paciente = st.text_input('Introduce el motivo de la visita')

# Botón para encontrar psicólogos
if st.button('Encontrar Psicólogos'):
    if edad_paciente and motivo_visita_paciente:
        top_3_psicologos_info = get_top_3_psychologists(edad_paciente.lower(), motivo_visita_paciente.lower())
        if not top_3_psicologos_info.empty:
            st.write("Los 3 psicólogos más similares son:")
            st.write(top_3_psicologos_info[['ID Psicólogo']])
        else:
            st.write("No se encontraron psicólogos con similitud significativa.")
    else:
        st.write("Por favor, introduce la edad y el motivo de la visita del paciente.")

num_points = st.slider("Number of points in spiral", 1, 10000, 1100)
num_turns = st.slider("Number of turns in spiral", 1, 300, 31)

indices = np.linspace(0, 1, num_points)
theta = 2 * np.pi * num_turns * indices
radius = indices

x = radius * np.cos(theta)
y = radius * np.sin(theta)

df = pd.DataFrame({
    "x": x,
    "y": y,
    "idx": indices,
    "rand": np.random.randn(num_points),
})

st.altair_chart(alt.Chart(df, height=700, width=700)
    .mark_point(filled=True)
    .encode(
        x=alt.X("x", axis=None),
        y=alt.Y("y", axis=None),
        color=alt.Color("idx", legend=None, scale=alt.Scale()),
        size=alt.Size("rand", legend=None, scale=alt.Scale(range=[1, 150])),
    ))
