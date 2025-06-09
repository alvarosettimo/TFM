# obtener datos de la API de Semantic Scholar

import requests
import time
import pandas as pd
import os

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,year,authors.name"  # no funciona el authors.name, ver. tampoco pude traerme el DOI, imp para citas
DATA_DIR = "C:\\Users\\alvar\\OneDrive\\Escritorio\\TFM\\data"
OUTPUT_FILE = os.path.join(DATA_DIR, "initial_corpus.csv")

def get_papers(query, limit=100):
    """Obtiene artículos de Semantic Scholar para una consulta dada."""
    papers_data = []
    offset = 0
    print(f"Buscando artículos para: '{query}'...")
    while offset < limit:
        try:
            params = {
                'query': query,
                'limit': min(100, limit - offset), # La API tiene un límite de 100 por página
                'offset': offset,
                'fields': FIELDS
            }
           
            response = requests.get(f"{SEMANTIC_SCHOLAR_API_URL}/paper/search", params=params)
            response.raise_for_status() # Lanza excepción si hay error HTTP
            data = response.json()

            # Si no hay datos o la clave 'data' no existe, parar
            if not data.get('data'):
                print("No se encontraron más datos.")
                break

            # Añadir solo artículos que tengan abstract
            count_added = 0
            for paper in data['data']:
                if paper.get('abstract'):
                    papers_data.append(paper)
                    count_added += 1
            print(f"  Añadidos {count_added} artículos con abstract de {len(data['data'])} encontrados en esta página.")

            offset += len(data['data'])

            # Comprobar si hay más páginas 
            # Paramos si el offset alcanza o supera el límite solicitado
            if data.get('next', 0) == 0 or offset >= limit:
                 print("Alcanzado el límite o no hay más páginas.")
                 break

            # Esperar para respetar los límites de la API (muy importante)
            # El límite sin autenticación es 100 peticiones / 5 minutos
            print(f"Esperando 3 segundos antes de la siguiente petición... (Total obtenidos hasta ahora: {len(papers_data)})")
            time.sleep(3)

        except requests.exceptions.RequestException as e:
            print(f"Error en la petición a la API: {e}")
            print("Esperando 10 segundos antes de reintentar o parar...")
            time.sleep(10)
            break
        except Exception as e:
            print(f"Error inesperado procesando datos: {e}")
            break

    print(f"Finalizada la búsqueda para '{query}'. Total de artículos con abstract obtenidos: {len(papers_data)}")
    return papers_data

if __name__ == "__main__":
    search_keywords = ["machine learning recommender system", "natural language processing academic search"]  # agregar aca todas las keywords a buscar
    papers_per_keyword = 150 # Intenta obtener hasta 150 por keyword 
    

    all_papers = []
    for keyword in search_keywords:
        papers = get_papers(keyword, limit=papers_per_keyword)
        all_papers.extend(papers)
        # Espera adicional entre diferentes keywords por las dudas
        print("Esperando 5 segundos antes de la siguiente keyword...")
        time.sleep(5)

    if not all_papers:
        print("No se obtuvieron artículos. Revisar las keywords o la conexión.")
    else:
        # Crear DataFrame y eliminar duplicados por paperId
        df = pd.DataFrame(all_papers)
        # Asegurarse de que la columna 'paperId' existe antes de eliminar duplicados
        if 'paperId' in df.columns:
            df.drop_duplicates(subset=['paperId'], inplace=True)
            df.reset_index(drop=True, inplace=True)
        else:
            print("Advertencia: La columna 'paperId' no se encontró en los datos recuperados. No se eliminaron duplicados.")

        # Asegurarse de que la carpeta de datos exista
        os.makedirs(DATA_DIR, exist_ok=True)

        # Guardar en CSV
        try:
            df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
            print(f"Total de {len(df)} artículos únicos con abstract guardados en '{OUTPUT_FILE}'")
        except Exception as e:
            print(f"Error al guardar el archivo CSV: {e}")