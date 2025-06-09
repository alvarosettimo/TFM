# MOTOR DE REOMENDACION

import numpy as np
import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer
import torch # Para la detección de GPU
import ast
import yake
import re

# Importar funciones de los otros módulos
from src.procesamiento_corpus import preprocess_text 
from src.calculo_similitud_coseno import calculate_cosine_similarities

# --- 1. Configuración y Carga de Recursos ---

# Nombre del modelo SBERT (TIENE QUE SER EL MISMO QUE EL DE LOS EMBEDDINGS DEL CORPUS)
SBERT_MODEL_NAME = 'all-MiniLM-L6-v2' 

# Rutas a los datos precalculados
BASE_DATA_PATH = 'C:\\Users\\alvar\\OneDrive\\Escritorio\\TFM\\data' 
EMBEDDINGS_FILE_NAME = f'corpus_embeddings_{SBERT_MODEL_NAME.replace("/", "_")}.npy'
IDS_FILE_NAME = f'corpus_paper_ids_{SBERT_MODEL_NAME.replace("/", "_")}.pkl'
METADATA_FILE_NAME = 'corpus_procesado.csv'

EMBEDDINGS_PATH = os.path.join(BASE_DATA_PATH, 'Embeddings', EMBEDDINGS_FILE_NAME)
IDS_PATH = os.path.join(BASE_DATA_PATH, 'Embeddings', IDS_FILE_NAME)
METADATA_PATH = os.path.join(BASE_DATA_PATH, 'Procesado', METADATA_FILE_NAME)

# Variables globales para los recursos cargados
sbert_model = None
corpus_embeddings = None
paper_ids_ordered = None
df_metadata = None
resources_loaded = False
paper_id_to_index = {}

def _load_resources():
    """Carga los recursos necesarios (modelo, embeddings, IDs, metadatos) una sola vez."""
    global sbert_model, corpus_embeddings, paper_ids_ordered, df_metadata, resources_loaded
    
    if resources_loaded:
        return True

    print("Cargando recursos para el motor de recomendación...")
    try:
        # Cargar modelo SBERT
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando dispositivo: {device} para SBERT.")
        sbert_model = SentenceTransformer(SBERT_MODEL_NAME, device=device)
        
        # Cargar embeddings del corpus
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(f"Archivo de embeddings no encontrado: {EMBEDDINGS_PATH}")
        corpus_embeddings = np.load(EMBEDDINGS_PATH)
        
        # Cargar IDs ordenados
        if not os.path.exists(IDS_PATH):
            raise FileNotFoundError(f"Archivo de IDs no encontrado: {IDS_PATH}")
        with open(IDS_PATH, 'rb') as f:
            paper_ids_ordered = pickle.load(f)
        
        # Crear el mapeo paper_id_to_index
        if paper_ids_ordered:
            paper_id_to_index = {pid: i for i, pid in enumerate(paper_ids_ordered)}
            
        # Cargar metadatos
        if not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(f"Archivo de metadatos no encontrado: {METADATA_PATH}")
        df_metadata = pd.read_csv(METADATA_PATH)
       
        if 'paperId' in df_metadata.columns:
             df_metadata.set_index('paperId', inplace=True, drop=False) # drop=False para mantener la columna

        print(f"Modelo SBERT '{SBERT_MODEL_NAME}' cargado.")
        print(f"Embeddings del corpus cargados. Forma: {corpus_embeddings.shape}")
        print(f"IDs del corpus cargados. Cantidad: {len(paper_ids_ordered)}")
        print(f"Metadatos del corpus cargados. Filas: {len(df_metadata)}")

        if not (corpus_embeddings.shape[0] == len(paper_ids_ordered) == len(df_metadata.drop_duplicates(subset=['paperId']))):
             print("ADVERTENCIA: El número de embeddings, IDs y metadatos únicos no coincide. Verificar los datos.")
        
        resources_loaded = True
        return True

    except FileNotFoundError as fnf_error:
        print(f"Error al cargar recurso: {fnf_error}")
        return False
    except Exception as e:
        print(f"error al cargar recursos: {e}")
        return False
    
# --- 1.1. Función para parsear una cadena de texto a un diccionario ---
def _parse_authors_string(authors_str: str) -> list:
    """Parsea una cadena que representa una lista de diccionarios de autores."""
    if pd.isna(authors_str) or not isinstance(authors_str, str):
        return []
    try:
        authors_list = ast.literal_eval(authors_str)
        if isinstance(authors_list, list):
            return authors_list
        return [] # No era una lista después de parsear
    except (ValueError, SyntaxError):
        return []

# --- 1.2. Función para formatear los detalles de una recomendación ---
def _format_recommendation_details(paper_id, score, rank_val):
    """Función auxiliar para formatear los detalles de una recomendación."""
    global df_metadata # Acceder al DataFrame global de metadatos
    try:
        paper_details = df_metadata.loc[paper_id]
        
        authors_data = paper_details.get('authors')
        author_names_list = []
        if isinstance(authors_data, list):
            author_names_list = [auth.get('name') for auth in authors_data if auth and isinstance(auth, dict) and auth.get('name')]
        elif isinstance(authors_data, str):
            parsed_list = _parse_authors_string(authors_data)
            author_names_list = [auth.get('name') for auth in parsed_list if auth and isinstance(auth, dict) and auth.get('name')]
        
        authors_display_string = "Autores no disponibles"
        if author_names_list:
            if len(author_names_list) > 3:
                authors_display_string = ", ".join(author_names_list[:3]) + ", et al."
            else:
                authors_display_string = ", ".join(author_names_list)

        return {
            'rank': rank_val,
            'paperId': paper_id,
            'title': paper_details.get('title', 'Título no disponible'),
            'abstract': paper_details.get('processed_text', paper_details.get('abstract', 'Abstract no disponible'))[:2000],
            'year': int(paper_details.get('year', 0)) if pd.notna(paper_details.get('year')) else 'Año no disponible',
            'authors': authors_display_string,
            'pdf_url': paper_details.get('pdf_url', None),
            'similarity_score': round(float(score), 4)
        }
    except KeyError:
        print(f"ADVERTENCIA: No se encontraron metadatos para el paperId: {paper_id}")
        return {
            'rank': rank_val,
            'paperId': paper_id,
            'title': 'Metadatos no encontrados',
            'abstract_snippet': 'N/A',
            'year': 'N/A',
            'authors': 'N/A',
            'pdf_url': None,
            'similarity_score': round(float(score), 4)
        }


# --- 2. Función Principal de Recomendación ---

def get_recommendations(user_query: str, top_n: int = 5) -> list[dict]:
    """
    Genera recomendaciones de recursos bibliograficos basadas en una consulta de usuario.

    Args:
        user_query (str): objetivo de estudio del usuario.
        top_n (int): cantidad de recomendaciones a devolver.

    Returns:
        list[dict]: Una lista de diccionarios, donde cada diccionario representa
                    un paper recomendado e incluye su 'paperId', 'title', 
                    'processed_text', 'year','pdf_url' y 'similarity_score'.
                    Devuelve lista vacía si hay errores o no hay resultados.
    """
    if not resources_loaded:
        success = _load_resources() # Intenta cargar recursos si no están listos
        if not success:
            print("Error: Los recursos no pudieron ser cargados. No se pueden generar recomendaciones.")
            return [] 

    if sbert_model is None or corpus_embeddings is None or paper_ids_ordered is None or df_metadata is None:
        print("Error: Faltan recursos esenciales para la recomendación.")
        return []

     # 1. Preprocesar consulta
    preprocessed_query = preprocess_text(user_query)
    if not preprocessed_query:
        # manejo de consulta vacía
        return []

    # 2. Embedding de la consulta
    query_embedding = sbert_model.encode([preprocessed_query], convert_to_numpy=True)
    
    # 3. Calcular similitudes
    similarity_scores = calculate_cosine_similarities(query_embedding, corpus_embeddings)
    
    # 4. Obtener top N índices
    effective_top_n = min(top_n, len(similarity_scores))
    top_n_indices = np.argsort(similarity_scores)[::-1][:effective_top_n]
    
    # 5. Recuperar detalles
    recommendations = []
    for rank, idx in enumerate(top_n_indices):
        paper_id = paper_ids_ordered[idx]
        score = similarity_scores[idx]
        recommendations.append(_format_recommendation_details(paper_id, score, rank + 1))
            
    return recommendations

# --- funcion para boton "mas como este" ---
def get_similar_to_paper(seed_paper_id: str, top_n: int = 5) -> list[dict]:
    """
    Encuentra papers similares a un paper "semilla" dado,
    recalculando el embedding del paper semilla a partir de su texto procesado.
    """
    global sbert_model, corpus_embeddings, paper_ids_ordered, df_metadata # Acceder a recursos globales

    if not resources_loaded:
        success = _load_resources()
        if not success:
            print("Error: Los recursos no pudieron ser cargados.")
            return [] 

    if sbert_model is None or corpus_embeddings is None or paper_ids_ordered is None or df_metadata is None:
        print("Error: Faltan recursos esenciales para la recomendación.")
        return []
        
    # 1. Obtener el texto procesado del paper semilla desde df_metadata
    try:
        paper_details_seed = df_metadata.loc[seed_paper_id]
        # Uso 'processed_text' porque es lo que se usó para generar corpus_embeddings
        text_to_embed_seed = paper_details_seed.get('processed_text') 
        
        if pd.isna(text_to_embed_seed) or not str(text_to_embed_seed).strip():
            print(f"Error: No se encontró texto procesado para el paperId semilla '{seed_paper_id}'.")
            return []
            
    except KeyError:
        print(f"Error: El paperId semilla '{seed_paper_id}' no se encontró en los metadatos (df_metadata).")
        return []

    # 2. Generar embedding para el texto del paper semilla en el momento
    print(f"Generando embedding para el paper semilla '{seed_paper_id}' sobre la marcha...")
    seed_embedding_vivo = sbert_model.encode([str(text_to_embed_seed)], convert_to_numpy=True)

    # 3. Calcular similitudes
    similarity_scores = calculate_cosine_similarities(seed_embedding_vivo, corpus_embeddings)

    # 4. Obtener los N MEJORES índices (no N+1, ya que el embedding del paper semilla no es necesariamente uno de los corpus_embeddings si se genera sobre la marcha)
  
    effective_top_n = min(top_n + 1, len(similarity_scores)) # Tomo uno extra por si el semilla aparece
    sorted_indices = np.argsort(similarity_scores)[::-1]

    # 5. Filtrar el paper semilla y tomar los top_n restantes
    recommendations = []
    rank_val = 1
    for idx in sorted_indices:
        if rank_val > top_n: 
            break
        
        current_paper_id = paper_ids_ordered[idx]
        
        # Evitar recomendar el mismo paper semilla
        if current_paper_id == seed_paper_id:
            continue 

        score = similarity_scores[idx]
        recommendations.append(_format_recommendation_details(current_paper_id, score, rank_val))
        rank_val += 1
            
    return recommendations



# --- Carga inicial de recursos ---
_load_resources()