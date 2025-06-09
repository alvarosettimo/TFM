# este script calcula la similitud del coseno entre un embedding de consulta y el  conjunto de embeddings del corpus

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarities(query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
    """
    Calcula las puntuaciones de similitud del coseno entre un embedding de consulta del usuario
    y un conjunto de embeddings del corpus.

    Args:
        query_embedding (np.ndarray): Un array de NumPy representando el embedding de la consulta.
                                      
        corpus_embeddings (np.ndarray): Un array de NumPy 2D donde cada fila es un embedding
                                        del corpus.
    
    Returns:
        np.ndarray: Un array de NumPy 1D con las puntuaciones de similitud del coseno
                    entre la consulta y CADA embedding del corpus.
    """
    # La función cosine_similarity de sklearn espera entradas 2D.
    # Si query_embedding es 1D, hay que convertirlo a 2D.
    if query_embedding.ndim == 1:
        query_embedding_2d = query_embedding.reshape(1, -1)
    elif query_embedding.ndim == 2 and query_embedding.shape[0] == 1:
        query_embedding_2d = query_embedding
    else:
        raise ValueError(f"query_embedding debe ser 1D o 2D con una sola fila. Forma actual: {query_embedding.shape}")

    if corpus_embeddings.ndim != 2:
        raise ValueError(f"corpus_embeddings debe ser un array 2D. Forma actual: {corpus_embeddings.shape}")

    # cosine_similarity devuelve un array 2D de forma (n muestras query, n muestras corpus)
    # Como solo hay una consulta, el resultado será (1, n_muestras_corpus)!!
    # Tomo la primera fila para obtener un array 1D de puntuaciones.
    sim_scores = cosine_similarity(query_embedding_2d, corpus_embeddings)
    
    return sim_scores[0] # esta variable tiene esta forma: [[puntuacion_consulta_vs_doc0, puntuacion_consulta_vs_doc1, puntuacion_consulta_vs_doc2, puntuacion_consulta_vs_doc3]]
    # sim_scores[0][i] va a tener la puntuacion de similitud entre la consulta y el embedding del i-esimo documento del corpus
    # sim_scores[0][i] = 1.0 significa que son idénticos, 0.0 significa que son ortogonales (sin similitud)