# generador de embeddings para el corpus

from sentence_transformers import SentenceTransformer
import numpy as np
import torch 

def generate_sbert_embeddings(text_list: list[str], 
                              model_name: str = 'all-MiniLM-L6-v2',
                              batch_size: int = 32,
                              show_progress: bool = True) -> np.ndarray:
    """
    Genera embeddings para una lista de textos usando un modelo SBERT especificado.

    Args:
        text_list: Lista de textos a codificar. (los abstracts procesados)
        model_name: Nombre del modelo SentenceTransformer a usar.('all-MiniLM-L6-v2', 'multi-qa-mpnet-base-dot-v1')
                          
        batch_size: Tamaño del lote para la codificación.
        show_progress: para mostrar una barra de progreso durante la codificación.

    Returns:
        np.ndarray: Un array de NumPy con los embeddings generados.
    """
    print(f"Cargando el modelo SBERT: {model_name}...")
    # Verificar si hay GPU disponible y usarla si es posible (para google colab)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    model = SentenceTransformer(model_name_or_path=model_name, device=device)
    
    print(f"Generando embeddings para {len(text_list)} textos...")
    embeddings = model.encode(
        text_list, 
        batch_size=batch_size, 
        show_progress_bar=show_progress,
        convert_to_numpy=True # Devuelve directamente un array NumPy
    )
    print(f"Embeddings generados con forma: {embeddings.shape}")
    return embeddings