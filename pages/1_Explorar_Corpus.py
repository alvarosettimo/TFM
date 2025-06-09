# Pagina para explorar el corpus bibliográfico

import streamlit as st
import pandas as pd
import os
import ast
from collections import Counter
from wordcloud import WordCloud, STOPWORDS as WORDCLOUD_BUILTIN_STOPWORDS
import matplotlib.pyplot as plt 
import plotly.express as px
from src.funciones_auxiliares import get_image_as_base64
import networkx as nx 
from itertools import combinations 
from streamlit_agraph import agraph, Node, Edge, Config 


# --- Configuración de la Página ---
st.set_page_config(
    page_title="Explorador del Corpus",
    page_icon="📚",
    layout="wide"
)

st.title("Explorador del Corpus Bibliográfico")
st.markdown("Aquí puedes explorar todos los documentos que forman parte de la base de conocimiento del recomendador.")

# --- Funciones Auxiliares ---
@st.cache_data
def _format_authors_display(authors_data):
    """
    Parsea y formatea los datos de autores para su visualización.
    authors_data puede ser una cadena representando una lista de diccionarios,
    o ya una lista de diccionarios.
    """
    if pd.isna(authors_data):
        return "N/A"

    author_names_list = []
    
    # Paso 1: Intentar parsear si es una cadena
    actual_list_of_dicts = []
    if isinstance(authors_data, str):
        try:
            if authors_data.strip().lower() == 'nan':
                 return "N/A"
            evaluated_data = ast.literal_eval(authors_data)
            if isinstance(evaluated_data, list):
                actual_list_of_dicts = evaluated_data
            else: # Si ast.literal_eval devuelve otra cosa que no es una lista
                return str(authors_data) # Devolver la cadena original si no es el formato esperado
        except (ValueError, SyntaxError):
            return str(authors_data) 
    elif isinstance(authors_data, list): 
        actual_list_of_dicts = authors_data
    else: # Si no es ni string ni lista
        return "Formato de autor desconocido"

    # Paso 2: Extraer nombres de la lista de diccionarios
    for author_dict in actual_list_of_dicts:
        if isinstance(author_dict, dict) and 'name' in author_dict:
            author_names_list.append(str(author_dict['name']))

    # Paso 3: Formatear la cadena de visualización
    if not author_names_list:
        # Si después de todo no hay nombres, pero la celda original no era NaN y era una lista vacía parseada
        if isinstance(actual_list_of_dicts, list) and not actual_list_of_dicts:
            return "Sin autores listados"
        # Si la celda original era una cadena que no se pudo interpretar como autores válidos y no produjo nombres
        if isinstance(authors_data, str) and authors_data.strip():
            return authors_data # Devolver la cadena original si no se pudo extraer nada específico
        return "N/A" # Caso por defecto
    
    if len(author_names_list) > 3:
        return ", ".join(author_names_list[:3]) + ", et al."
    else:
        return ", ".join(author_names_list)
    
    
@st.cache_data   
def get_top_authors(df_column_authors, top_n=10):
    all_authors_list = []
    if df_column_authors.empty: return pd.DataFrame(columns=['Autor', 'Número de Artículos']) # Devolver DF vacío
    for entry in df_column_authors.dropna():
        authors_in_entry = []
        if isinstance(entry, str):
            try:
                if entry.strip().lower() == 'nan': continue
                list_of_dicts = ast.literal_eval(entry)
                if isinstance(list_of_dicts, list):
                    authors_in_entry = [str(d.get('name')) for d in list_of_dicts if isinstance(d, dict) and d.get('name')]
            except (ValueError, SyntaxError):
                authors_in_entry = [name.strip() for name in str(entry).split(',') if name.strip()]
        elif isinstance(entry, list):
             authors_in_entry = [str(d.get('name')) for d in entry if isinstance(d, dict) and d.get('name')]
        all_authors_list.extend(auth_name for auth_name in authors_in_entry if auth_name and auth_name.lower() != 'nan' and auth_name.lower() != 'n/a')

    if not all_authors_list: return pd.DataFrame(columns=['Autor', 'Número de Artículos'])

    author_counts = Counter(all_authors_list)
    # Devolver un DataFrame con 'Autor' y 'Número de Artículos' como columnas
    top_authors_df = pd.DataFrame(author_counts.most_common(top_n), columns=['Autor', 'Número de Artículos'])
    return top_authors_df


@st.cache_data
def generate_wordcloud_from_texts(texts_list: list, additional_custom_stopwords: set = None): 
    """Genera un objeto WordCloud a partir de una lista de textos."""
    if not texts_list:
        return None
    
    # Unir todos los textos en una sola cadena y convertir a minúsculas
    text = " ".join(str(t).lower() for t in texts_list if pd.notna(t) and str(t).strip())
    
    if not text: # Si después de unir y limpiar no hay texto
        return None

    # 1.lista de stopwords incorporada en la biblioteca wordcloud
    current_stopwords = set(WORDCLOUD_BUILTIN_STOPWORDS)
    
    # 2.stopwords comunes del ambito académico/investigación
    domain_specific_stopwords = {
        "paper", "study", "article", "research", "title", "review", 
        "based", "using", "towards", "approach", "et", "al", "fig", "figure", "table",
        "abstract", "keywords", "keyword", "term", "terms", "section", "chapter",
        "cid", "objective", "background", "method", "introduction", "conclusion", "results",
        "vol", "issue", "page", "editor", "publisher", "university", "conference", "journal","of","al","the","a","in"
    }
    current_stopwords.update(domain_specific_stopwords)

    wordcloud = WordCloud(width=1000, height=500, # tamaño de la imagen
                          background_color='white',
                          stopwords=current_stopwords, # Usar el conjunto final de stopwords
                          min_font_size=10,
                          max_words=150, 
                          collocations=False # Evitar que cuente bigramas comunes como una sola palabra
                         ).generate(text) 
    return wordcloud


@st.cache_data  
def get_authors_per_paper_distribution(df_column_authors):
    """
    Calcula la distribución del número de autores por paper.
    Espera una columna de DataFrame donde cada celda puede ser:
    - NaN
    - Una cadena representando una lista de diccionarios (ej. "[{'name': 'A'}, {'name': 'B'}]")
    - Una lista de diccionarios (si ya fue parseada)
    """
    num_authors_list = []
    if df_column_authors.empty:
        return pd.Series(dtype='int')

    for entry in df_column_authors.dropna(): # Ignorar NaNs directamente
        count_for_entry = 0
        authors_in_entry = []
        if isinstance(entry, str):
            try:
                if entry.strip().lower() == 'nan': continue
                list_of_dicts = ast.literal_eval(entry)
                if isinstance(list_of_dicts, list):
                    authors_in_entry = [d for d in list_of_dicts if isinstance(d, dict) and d.get('name')]
            except (ValueError, SyntaxError):
                if str(entry).strip(): # Si la cadena original no estaba vacía
                    authors_in_entry = [{'name': str(entry).strip()}] # Asumir que es un solo nombre o un grupo
        elif isinstance(entry, list): # Si ya es una lista
            authors_in_entry = [d for d in entry if isinstance(d, dict) and d.get('name')]
        
        count_for_entry = len(authors_in_entry)
        if count_for_entry > 0:
            num_authors_list.append(count_for_entry)

    if not num_authors_list:
        return pd.Series(dtype='int')
        
    # Contar la frecuencia de cada número de autores
    distribution = pd.Series(num_authors_list).value_counts().sort_index()
    return distribution



# --- Carga de Datos ---
DATA_DIR_ROOT_RELATIVE = "data" 
METADATA_FILE_NAME = 'corpus_procesado.csv'
METADATA_PATH = os.path.join(DATA_DIR_ROOT_RELATIVE, 'Procesado', METADATA_FILE_NAME)

@st.cache_data
def load_data(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Limpieza básica y formateo para visualización
            df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
            df['title'] = df['title'].fillna('N/A')
            df['pdf_url'] = df['pdf_url'].fillna('')
            df['abstract'] = df['abstract'].fillna('N/A') 

            # Aplicar el formateo a la columna 'authors' para crear una columna de visualización
            if 'authors' in df.columns:
                df['authors_display'] = df['authors'].apply(_format_authors_display)
            else:
                df['authors_display'] = "N/A" # Columna por defecto si 'authors' no existe
            
            return df
        except Exception as e:
            st.error(f"Error al cargar o procesar el archivo de metadatos: {e}")
            return pd.DataFrame()
    else:
        st.warning(f"Archivo de metadatos no encontrado en: {path}")
        return pd.DataFrame()

df_corpus_loaded = load_data(METADATA_PATH)


if not df_corpus_loaded.empty:
    st.info(f"Cantidad total de documentos: {len(df_corpus_loaded)}")

    # --- Filtros ---
    logo_path = "C:\\Users\\alvar\\OneDrive\\Escritorio\\TFM\\static\\logo_app2.png"  
    logo_base64_encoded = get_image_as_base64(logo_path)
    st.sidebar.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin-bottom: 5px;">
                    <img src="data:image/png;base64,{logo_base64_encoded}" alt="Logo App" width="100" style="position: relative; left: -15px;">
                </div>
                """,
                unsafe_allow_html=True)
    st.sidebar.header("Filtros del Corpus")
    search_term = st.sidebar.text_input("Buscar por Título/Autores:", placeholder="Escribe para filtrar...")

    if 'year' in df_corpus_loaded.columns and not df_corpus_loaded['year'].empty:
        min_corpus_year = int(df_corpus_loaded['year'].min())
        max_corpus_year = int(df_corpus_loaded['year'].max())
        if min_corpus_year < max_corpus_year:
             selected_year_range = st.sidebar.slider(
                "Filtrar por Año de Publicación:",
                min_value=min_corpus_year,
                max_value=max_corpus_year,
                value=(min_corpus_year, max_corpus_year)
            )
        else:
            selected_year_range = (min_corpus_year, max_corpus_year)
            st.sidebar.caption(f"Todos los documentos son del año {min_corpus_year}")
    else: # Fallback si no hay columna 'year' o está vacía
        selected_year_range = None


    # Aplicar filtros
    df_filtered = df_corpus_loaded.copy()

    if search_term: 
        df_filtered = df_filtered[
            df_filtered['title'].str.contains(search_term, case=False, na=False) |
            df_filtered['authors_display'].str.contains(search_term, case=False, na=False) 
        ]
    
    if selected_year_range and 'year' in df_filtered.columns and min_corpus_year < max_corpus_year:
        df_filtered = df_filtered[
            (df_filtered['year'] >= selected_year_range[0]) &
            (df_filtered['year'] <= selected_year_range[1])
        ]

    
    st.write(f"Mostrando {len(df_filtered)} documentos después de aplicar filtros.")

    # --- Visualización de la Tabla ---
    st.subheader("Detalle del Corpus")
    columns_to_display = {
        'title': 'Título',
        'authors_display': 'Autores',
        'year': 'Año',
        'pdf_url': 'Enlace PDF',
        'abstract': 'Resumen'
    }
    df_display = df_filtered[list(columns_to_display.keys())].copy()
    df_display.rename(columns=columns_to_display, inplace=True)
    
    st.data_editor(
        df_display,
        column_config={
            "Enlace PDF": st.column_config.LinkColumn(
                "Enlace PDF",
                help="Haz clic para abrir el PDF (si está disponible)", # Texto que se muestra cuando te paras arriba de la columna
                display_text="🔗 Abrir PDF", 
            ),
            "Título": st.column_config.TextColumn(width="large"),
            "Autores": st.column_config.TextColumn(width="medium") 
        },
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic"
    )

# --- GRAFICAS ---
    if not df_filtered.empty:
        st.markdown("---") # Separador antes de la sección de gráficos
        st.subheader("Analítica Visual del Corpus")

        # --- Gráfico de Artículos por Año ---
        st.markdown("##### Distribución de Artículos por Año")
        year_counts = df_filtered[df_filtered['year'] != 0]['year'].value_counts().sort_index()
        if not year_counts.empty:
            chart_data_year = pd.DataFrame({'Año': year_counts.index.astype(int), 'Número de Artículos': year_counts.values}).sort_values(by='Año')
            st.bar_chart(chart_data_year.set_index('Año'))
        else:
            st.caption("No hay datos de año para mostrar en el gráfico con los filtros actuales.")
        
        # --- Gráfico de top N autores ---
        st.markdown("##### Top 10 Autores mas prolificos")
        if 'authors' in df_filtered.columns:
            top_authors_df_for_plotly = get_top_authors(df_filtered['authors'], top_n=10)

            if not top_authors_df_for_plotly.empty:
                import plotly.express as px 
                fig_top_authors = px.bar(
                    top_authors_df_for_plotly.sort_values(by="Número de Artículos", ascending=True), # Ordenar para que la barra más larga quede arriba
                    x="Número de Artículos",
                    y="Autor",
                    orientation='h', # Esto lo hace horizontal
                    height=max(400, len(top_authors_df_for_plotly) * 40) # Altura dinámica
                )
                # Ajustes opcionales de layout
                fig_top_authors.update_layout(
                    yaxis_title="", 
                    xaxis_title="Número de Artículos"
                )
                st.plotly_chart(fig_top_authors, use_container_width=True)
            else:
                st.caption("No hay suficientes datos de autores para mostrar este gráfico con los filtros actuales.")
        else:
            st.caption("Columna 'authors' no encontrada para generar este gráfico.")
        
        # --- wordcloud de palabras claves en los titulos ---
        st.markdown("##### Nube de Palabras Clave en Títulos")
        if 'title' in df_filtered.columns:
            titles_for_wc = df_filtered['title'].dropna().tolist()
            if titles_for_wc:
                wordcloud_object = generate_wordcloud_from_texts(titles_for_wc) 
                if wordcloud_object:
                    fig, ax = plt.subplots(figsize=(12, 6)) 
                    ax.imshow(wordcloud_object, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.caption("No hay suficiente texto en los títulos para generar la nube de palabras con los filtros actuales.")
            else:
                st.caption("No hay títulos disponibles para generar la nube de palabras.")
        else:
            st.caption("Columna 'title' no encontrada para generar la nube de palabras.")
        
        # --- Gráfico de Distribución de Autores por Artículo ---
        st.markdown("##### Distribución de Autores por Artículo")
        if 'authors' in df_filtered.columns: # Usar la columna 'authors' original
            author_counts_dist = get_authors_per_paper_distribution(df_filtered['authors'])
            if not author_counts_dist.empty:
                st.bar_chart(author_counts_dist)
                
            else:
                st.caption("No hay suficientes datos de autores para mostrar este gráfico con los filtros actuales.")
        else:
            st.caption("Columna 'authors' no encontrada para generar este gráfico.")
        
        # --- Cantidad de pdfs disponibles ---
        st.markdown("##### Disponibilidad de PDFs de Acceso Abierto")
        if 'pdf_url' in df_filtered.columns:
            # Contar URLs válidas (no vacías y que parecen URLs) vs. no válidas/vacías
            has_valid_pdf = df_filtered['pdf_url'].apply(lambda x: isinstance(x, str) and x.startswith('http'))
            pdf_availability_counts = has_valid_pdf.value_counts()
            
            # Renombrar el índice para mayor claridad en el gráfico
            pdf_availability_counts.index = pdf_availability_counts.index.map({
                True: 'Con PDF Accesible', 
                False: 'Sin PDF / No Válido'
            })
            if not pdf_availability_counts.empty:
                fig_pie = px.pie(pdf_availability_counts, values=pdf_availability_counts.values, names=pdf_availability_counts.index)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.caption("No hay datos de URL de PDF para mostrar este gráfico con los filtros actuales.")
        else:
            st.caption("Columna 'pdf_url' no encontrada para generar este gráfico.")
        
else:
    st.error("No se pudo cargar la información del corpus. Verifica la ruta del archivo o el archivo mismo.")


# --- Información Adicional o Pie de Página ---
st.sidebar.markdown("---")
st.sidebar.info(
    "El corpus bibliografico fue construido utilizando la API de Semantic Scholar y "
    "contiene documentos de diversas areas de investigacion como IA, NLP, medicina, robotica, etc."
)