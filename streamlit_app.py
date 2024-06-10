import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data
data = {
    'ID Psicólogo': [37, 52, 38, 28, 1, 10, 61, 66, 62, 73, 13, 32, 58, 14, 54, 21, 24, 71, 57, 64, 46, 22, 9, 67, 27, 65, 70, 39, 50, 29, 51, 40, 35, 53, 72, 8, 56, 18, 34, 20, 26, 55, 2, 60, 15, 33, 49, 3, 11, 31, 47, 12, 7, 5, 30, 6, 59, 45, 44],
    'Edad paciente': [18.0, 33.0, 38.0, 28.0, 23.0, 28.0, 28.0, 18.0, 43.0, 28.0, 33.0, 12.0, 38.0, 38.0, 18.0, 53.0, 33.0, 38.0, 60.5, 43.0, 28.0, 53.0, 28.0, 33.0, 12.0, 28.0, 23.0, 33.0, 60.5, 48.0, 43.0, 28.0, 33.0, 23.0, 33.0, 38.0, 23.0, 53.0, 38.0, 43.0, 28.0, 33.0, 18.0, 28.0, 12.0, 28.0, 18.0, 12.0, 28.0, 23.0, 38.0, 23.0, 23.0, 38.0, 23.0, 28.0, 48.0, 33.0, 23.0],
    'Motivo visita paciente': ['me encuentro mal de salud sobretodo la barriga y no sé qué tengo', 'problemas de ansiedad', 'psicología perinatal', 'ruptura o celos', 'estado depresivo', 'siento que no valgo', 'tristeza', 'depresión', 'no siento ganas de nada', 'perfeccionismo e inseguridades', 'gestión relacional', 'mejoría en habilidades sociales ', 'divorcio', 'tener ansiedad', 'problemas emocionales', 'depresión ', 'problemas en relaciones ', 'dificultades familiares y depresión', 'tristeza profunda', 'dificultades en la relación con la madre', 'ansiedad y pensamientos intrusivos', 'angustia vital de vida ', 'dejar de sentir ansiedad', 'malestar emocional ', 'no queria comer para no engordar', 'residía fuera de su ciudad y tenía problemas en el trabajo', 'malestar', 'duelo pareja y relación tóxica ', 'me ahogo', 'ansiedad', 'estoy muy bajo de ánimos /energia', 'trauma, relaciones familiares', 'terapia de pareja, ella tenía dolor en la penetración', 'depresión ', 'problemas familiares ', 'ansiedad', 'falta de deseo sexual ', 'depresion', 'ansiedad', 'dificultades para relacionarse con personas, con mucha ansiedad', 'estrés desmedido por su trabajo', 'traumas relacionados con la pareja', 'ansiedad', 'malestar, ansiedad', 'tca', 'malestar ', 'ansiedad', 'desmotivación', 'ansiedad', 'problemas gestión emociones y alimentación ', 'ansiedad ', 'ansiedad', 'tca y suicidio', 'ansiedad', 'estrés por su situación laboral', 'depresión ', 'dejar de agobiarse por todo ', 'mejorar la relación conmigo misma ', 'autoimagen']
}

# Function to process the data and perform TF-IDF vectorization
def process_data(data):
    df = pd.DataFrame(data)
    df = df.applymap(lambda s: s.lower() if isinstance(s, str) else s)
    df['texto_combinado'] = df['Edad paciente'].astype(str) + ' ' + df['Motivo visita paciente']
    
    # Vectorize the texts using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['texto_combinado'])
    
    return df, vectorizer, tfidf_matrix

# User interface
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFF0B5;
    }
    </style>
    """,
    unsafe_allow_html=True
)
#st.image("https://drive.google.com/file/d/1h2jfotBC7DRnz25IDWLRgNMY1Bc3qBQN/view?usp=share_link", caption="Logo Calimana", use_column_width=True)
st.title('Recomendación de Psicólogos')

edad_input = st.text_input('Ingrese la edad del paciente:')
motivo_input = st.text_input('Ingrese el motivo de la visita:')

if st.button('Buscar Psicólogos'):
    if edad_input and motivo_input:
        df, vectorizer, tfidf_matrix = process_data(data)
        
        # Create the patient input
        patient_info = f"{edad_input} {motivo_input}"
        patient_vector = vectorizer.transform([patient_info])
        
        # Calculate cosine similarity
        similarity_scores = cosine_similarity(patient_vector, tfidf_matrix)
        
        # Filter and sort the results
        filtered_scores = [(index, score) for index, score in enumerate(similarity_scores[0]) if score > 0]
        top_matches = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:3]

        # Show the results
        if top_matches:
            st.write("Los 3 psicólogos más similares son:")
            for index, score in top_matches:
                st.write(f"ID Psicólogo: {df.iloc[index]['ID Psicólogo']} - Similitud: {score:.2f}")
        else:
            st.write("No se encontraron psicólogos con una similitud mayor a 0.")
    else:
        st.write("Por favor, ingrese tanto la edad del paciente como el motivo de la visita.")

