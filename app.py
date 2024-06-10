import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Función para cargar los datos
def load_data(file_path):
    df = pd.read_excel(file_path)
    df = df.applymap(lambda s: s.lower() if type(s) == str else s)
    df['texto_combinado'] = df['Edad paciente'].astype(str) + ' ' + df['Motivo visita paciente']
    return df

# Cargar los datos
df = load_data("answ.xlsx")

# Vectorizar los textos utilizando TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['texto_combinado'])

# Interfaz de usuario de Streamlit
st.title('Recomendación de Psicólogos')
edad_input = st.text_input('Ingrese la edad del paciente:')
motivo_input = st.text_input('Ingrese el motivo de la visita:')

if st.button('Buscar Psicólogos'):
    if edad_input and motivo_input:
        # Crear la entrada del paciente
        patient_info = f"{edad_input} {motivo_input}"
        patient_vector = vectorizer.transform([patient_info])
        
        # Calcular la similitud coseno
        similarity_scores = cosine_similarity(patient_vector, tfidf_matrix)
        
        # Filtrar y ordenar los resultados
        filtered_scores = [(index, score) for index, score in enumerate(similarity_scores[0]) if score > 0]
        top_matches = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:3]

        # Mostrar los resultados
        if top_matches:
            st.write("Los 3 psicólogos más similares son:")
            for index, score in top_matches:
                st.write(f"ID Psicólogo: {df.iloc[index]['ID Psicólogo']} - Similitud: {score:.2f}")
        else:
            st.write("No se encontraron psicólogos con una similitud mayor a 0.")
    else:
        st.write("Por favor, ingrese tanto la edad del paciente como el motivo de la visita.")
