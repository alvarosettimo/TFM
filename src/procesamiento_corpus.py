# Procesamiento datos del corpus para SBERT
# Este script se encarga de preprocesar el corpus inicial para que esté listo

import pandas as pd
import re
import numpy as np #para manejar NaN si es necesario
import ast 

# --- PASO 0: Carga de datos ---

df_corpus = pd.read_csv("C:\\Users\\alvar\\OneDrive\\Escritorio\\TFM\\data\\crudo\\initial_corpus.csv")


# --- PASO 1: Funciones de Preprocesamiento ---

def extract_pdf_url_from_dict_string(dict_string):
    if pd.isna(dict_string) or not isinstance(dict_string, str):
        return None
    try:
        # Convertir la cadena a un diccionario Python real
        data_dict = ast.literal_eval(dict_string)
        if isinstance(data_dict, dict):
            return data_dict.get('url', None) # Obtener el valor de la clave 'url'
        else:
            return None # No era un diccionario tras la evaluación
    except (ValueError, SyntaxError):
        # La cadena no era un literal de diccionario Python válido
        print(f"Advertencia: No se pudo parsear la cadena: {dict_string[:100]}...") # Muestra solo una parte
        return None

def remove_html_tags(text):
  """Elimina etiquetas HTML básicas de un texto."""
  if pd.isna(text):
      return None
  clean = re.compile('<.*?>')
  return re.sub(clean, '', text)

def normalize_whitespace(text):
  """Normaliza los espacios en blanco (múltiples espacios/saltos -> un espacio)."""
  if pd.isna(text):
      return None
  # Reemplaza cualquier secuencia de espacios/saltos/tabs por un solo espacio
  text = re.sub(r'\s+', ' ', text)
  # Elimina espacios al principio y al final
  return text.strip()

def preprocess_text(text):
  """Aplica la secuencia completa de preprocesamiento a un único texto."""
  if pd.isna(text):
    return None

  text = str(text) # Asegurarse de que es string
  text = text.lower() # Convertir a minúsculas
  text = remove_html_tags(text) # Eliminar HTML
  text = normalize_whitespace(text) # Normalizar espacios

  # Devolver None si el texto queda vacío después de limpiar
  return text if text else None



# --- PASO 2: Selección del Texto Fuente (Abstract o Título) ---

def get_text_source(row, min_abstract_len=50):
  """Decide si usar abstract o title basado en disponibilidad y longitud."""
  abstract = row['abstract']
  title = row['title']
  processed_abstract = None
  processed_title = None

  # Intenta procesar el abstract
  if not pd.isna(abstract):
      temp_abstract = preprocess_text(abstract)
      if temp_abstract and len(temp_abstract.split()) >= min_abstract_len:
          processed_abstract = temp_abstract

  # Si el abstract no es usable, intenta procesar el título
  if processed_abstract is None and not pd.isna(title):
      processed_title = preprocess_text(title)
      # No aplico longitud mínima al título, ya que es la última opción

  # Devuelve el texto seleccionado
  return processed_abstract if processed_abstract is not None else processed_title


# --- PASO 3: Aplicar el Preprocesamiento al DataFrame ---

# Aplica la función 'get_text_source' a cada fila del DataFrame
# Esto crea una nueva columna 'processed_text' con el texto listo para SBERT
df_corpus['processed_text'] = df_corpus.apply(lambda row: get_text_source(row, min_abstract_len=50), axis=1)

print("\n--- Corpus con Texto Preprocesado ('processed_text') ---")
print(df_corpus[['paperId', 'title', 'abstract', 'processed_text']])
print("-" * 40)

# --- PASO 4: Manejar Artículos Sin Texto Útil ---
# Revisa cuántos artículos quedaron sin texto procesado útil

original_count = len(df_corpus)
df_corpus.dropna(subset=['processed_text'], inplace=True) # Elimina filas donde 'processed_text' es None/NaN


final_count = len(df_corpus)

print(f"\nSe procesaron {original_count} artículos.")
if original_count > final_count:
    print(f"Se eliminaron {original_count - final_count} artículos por no tener abstract ni título útil tras preprocesar.")
print(f"Corpus final para SBERT contiene {final_count} artículos.")
print("-" * 40)

# Ahora df_corpus contiene la columna 'processed_text' lista para ser usada, los embeddings de SBERT se van a generar sobre esta columna


# ver si conviene guardar este DataFrame procesado:
# df_corpus.to_csv('corpus_procesado.csv', index=False)
# df_corpus.to_json('corpus_procesado.json', orient='records', lines=True)


