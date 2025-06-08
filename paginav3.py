import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import re
import os

# =======================
# CARGA DE DATOS
# =======================
# Leer X_lda desde el archivo TXT
X_lda = np.loadtxt("X_lda.txt", delimiter=",", skiprows=1)  # Saltar la fila del encabezado
merged_df = pd.read_csv("merged_df_filtrado.csv")

merged_df["imdbId"] = merged_df["imdbId"].astype(str)
# Convertir valores no numéricos a NaN
merged_df["movieId"] = pd.to_numeric(merged_df["movieId"], errors="coerce")

# Reemplazar NaN con un valor válido (por ejemplo, 0)
merged_df["movieId"] = merged_df["movieId"].fillna(0).astype(float).astype(int).astype(str)

# =======================
# a) BÚSQUEDA POR SIMILITUD VISUAL
# =======================
# Buscar películas similares a una película con un imdbId específico
def buscar_similares(id, n=5, metodo="imdbId"):
    """
    Busca películas similares a una dada basada en su imdbId o movieId.
    :param id: El identificador de la película (imdbId o movieId).
    :param n: Número de películas similares a devolver.
    :param metodo: Método de búsqueda ("imdbId" o "movieId").
    :return: DataFrame con las películas similares.
    """
    if metodo not in ["imdbId", "movieId"]:
        raise ValueError("El método de búsqueda debe ser 'imdbId' o 'movieId'.")
    index_pelicula = merged_df[merged_df[metodo] == str(id)].index[0]
    nn = NearestNeighbors(n_neighbors=n+1).fit(X_lda)
    dists, idxs = nn.kneighbors([X_lda[index_pelicula]])
    columnas_necesarias = ["Title", "Genre", "Year", "Poster"]
    return merged_df.iloc[idxs[0][1:]][columnas_necesarias]

def filtrar_peliculas(genero=None, anio=None):
    df = merged_df.copy()
    if genero:
        df = df[df["Genre"].str.contains(genero, case=False, na=False)]
    if anio:
        df = df[df["Year"] == str(anio)]  # Filtrar por año como string
    columnas = ["Title", "Genre", "Poster", "cluster", "x", "y", "Year", "imdbId", "movieId"]
    return df[columnas]

# =======================
# STREAMLIT APP
# =======================
st.set_page_config(page_title="Recomendador de Películas", layout="centered")

# Inicializar variables en st.session_state
if "pagina_actual" not in st.session_state:
    st.session_state.pagina_actual = 1

if "mostrar_recomendaciones" not in st.session_state:
    st.session_state.mostrar_recomendaciones = False

if "pelicula_seleccionada" not in st.session_state:
    st.session_state.pelicula_seleccionada = None

# Título centrado y en rojo
st.markdown(
    """
    <h1 style="text-align: center; color: red; font-weight: bold;">
         Movies Plus
    </h1>
    """,
    unsafe_allow_html=True
)

# Texto más grande y en negrita
if not st.session_state.mostrar_recomendaciones:
    st.markdown(
        """
        <h2 style="text-align: center; color: black; font-weight: bold;">
            🎬 Catálogo completo de películas
        </h2>
        """,
        unsafe_allow_html=True
    )

    # =======================
    # FILTROS
    # =======================
    # Extraer géneros únicos
    generos_unicos = set()
    merged_df["Genre"].dropna().apply(lambda x: generos_unicos.update(x.split("|")))
    generos_unicos = sorted(generos_unicos)

    # Selección de género
    genero_seleccionado = st.selectbox(
        "Filtrar por género:",
        options=["Todos"] + generos_unicos,
        index=0
    )

    # Selección de año
    anio_seleccionado = st.selectbox(
        "Filtrar por año:",
        options=["Todos"] + sorted(merged_df["Year"].dropna().unique()),
        index=0
    )

    # Aplicar filtros
    if genero_seleccionado != "Todos" or anio_seleccionado != "Todos":
        catalogo_filtrado = filtrar_peliculas(
            genero=genero_seleccionado if genero_seleccionado != "Todos" else None,
            anio=anio_seleccionado if anio_seleccionado != "Todos" else None
        )
    else:
        catalogo_filtrado = merged_df

    # =======================
    # PAGINACIÓN
    # =======================
    # Número de pósters por página
    POSTERS_POR_PAGINA = 20

    # Botones para cambiar de página
    col1, col2 = st.columns([4, 1])  # Ajustar proporción de columnas para mover el botón hacia la izquierda
    with col1:
        if st.button("⬅️ Anterior"):
            st.session_state.pagina_actual = max(1, st.session_state.pagina_actual - 1)
    with col2:
        if st.button("➡️ Siguiente"):
            st.session_state.pagina_actual += 1

    # Calcular el rango de películas a mostrar según la página actual
    inicio = (st.session_state.pagina_actual - 1) * POSTERS_POR_PAGINA
    fin = inicio + POSTERS_POR_PAGINA
    catalogo_pagina = catalogo_filtrado.iloc[inicio:fin]

    # Mostrar texto de la página centrado encima de las películas
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 24px; font-weight: bold; color: black; margin-bottom: 20px;">
            Página {st.session_state.pagina_actual}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mostrar pósters de la página actual con botones debajo
    cols = st.columns(5)
    for idx, (title, genre, poster, imdbId) in enumerate(catalogo_pagina[["Title", "Genre", "Poster", "imdbId"]].values):
        with cols[idx % 5]:
            st.image(poster, width=120, caption=f"{title}")
            st.write(f"Género: {genre}")
            
            # Botón para ver recomendaciones
            if st.button(f"Ver similares", key=f"recomendaciones_{imdbId}"):
                st.session_state.mostrar_recomendaciones = True
                st.session_state.pelicula_seleccionada = imdbId

    # =======================
    # TOP 10 PELÍCULAS POR CLUSTER
    # =======================
    st.markdown(
        """
        <h2 style="text-align: center; color: black; font-weight: bold;">
            🔝 Top Películas Representativas por Cluster
        </h2>
        """,
        unsafe_allow_html=True
    )

    # Selección de cluster
    clusters_unicos = sorted(merged_df["cluster"].dropna().unique())
    cluster_seleccionado = st.selectbox(
        "Selecciona un cluster:",
        options=clusters_unicos,
        index=0
    )

    # Mostrar el top 10 de películas del cluster seleccionado
    peliculas_cluster = merged_df[merged_df["cluster"] == cluster_seleccionado].head(10)  # Seleccionar las primeras 10 películas
    cols = st.columns(5)
    for idx, (title, genre, year, poster) in enumerate(peliculas_cluster[["Title", "Genre", "Year", "Poster"]].values):
        with cols[idx % 5]:
            st.image(poster, width=120, caption=f"{title} ({year})")
            st.write(f"Género: {genre}")

else:
    # Mostrar recomendaciones si se seleccionó una película
    pelicula_seleccionada = merged_df[merged_df["imdbId"] == st.session_state.pelicula_seleccionada]
    if not pelicula_seleccionada.empty:
        st.markdown(
            """
            <h2 style="text-align: center; color: blue; font-weight: bold;">
                🎥 Película seleccionada
            </h2>
            """,
            unsafe_allow_html=True
        )
        poster = pelicula_seleccionada.iloc[0]["Poster"]
        title = pelicula_seleccionada.iloc[0]["Title"]
        year = pelicula_seleccionada.iloc[0]["Year"]
        genre = pelicula_seleccionada.iloc[0]["Genre"]

        # Mostrar la película seleccionada centrada
        st.markdown(
            f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="{poster}" alt="{title}" style="width: 300px; border-radius: 10px;">
                <h3 style="color: black; font-weight: bold;">{title} ({year})</h3>
                <p style="color: gray; font-size: 18px;">Género: {genre}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Mostrar las recomendaciones debajo
    st.markdown(
        """
        <h2 style="text-align: center; color: green; font-weight: bold;">
            Recomendaciones
        </h2>
        """,
        unsafe_allow_html=True
    )
    recomendaciones = buscar_similares(st.session_state.pelicula_seleccionada, n=15, metodo="imdbId")
    cols = st.columns(5)
    for idx, (title, genre, year, poster) in enumerate(recomendaciones.values):
        with cols[idx % 5]:
            st.image(poster, width=120, caption=f"{title} ({year})")
            st.write(f"Género: {genre}")

    # Botón para regresar a la página principal
    if st.button("Volver al catálogo"):
        st.session_state.mostrar_recomendaciones = False
        st.session_state.pelicula_seleccionada = None
