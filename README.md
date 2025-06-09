# Recomendador Bibliográfico con NLP Semántico

Este proyecto es el Trabajo de Fin de Máster de Ciencia de Datos en Universidad Europea de Madrid. Consiste en el diseño y desarrollo de una aplicación web que recomienda recursos bibliográficos (principalmente artículos científicos) basándose en la similitud semántica de un "objetivo de estudio" formulado por el usuario en lenguaje natural.

## Descripción del Proyecto

La aplicación busca solucionar el problema de la sobrecarga de información en la investigación académica, ofreciendo una alternativa a los buscadores tradicionales basados en palabras clave. El sistema utiliza embeddings semánticos generados por modelos Transformer (SBERT) para entender la intención detrás de la consulta del usuario y encontrar los artículos más relevantes en un corpus de documentos.

## Características Principales

- **Recomendación Semántica:** Utiliza `Sentence-BERT` para generar embeddings y la similitud coseno para rankear los resultados.
- **Interfaz de Usuario Interactiva:** Desarrollada con Streamlit, incluye una página de recomendaciones y un explorador del corpus, protegidas por un sistema de login.
- **Explorador de Corpus:** Permite visualizar, filtrar y analizar la base de datos de artículos mediante gráficos interactivos (artículos por año, top autores, nube de palabras, etc.).
- **Funcionalidades Avanzadas:** Incluye "Encontrar más como este" para profundizar en un tema y botones de feedback de relevancia.

## Tecnologías Utilizadas

- **Lenguaje:** Python 3.9+
- **Bibliotecas Principales:**
  - **Streamlit:** Para la interfaz de usuario web.
  - **Pandas:** Para la manipulación de datos.
  - **Sentence-Transformers (Hugging Face):** Para la generación de embeddings semánticos con SBERT.
  - **Scikit-learn:** Para el cálculo de similitud coseno.
  - **BERTopic:** Para el modelado de tópicos y generación del ground truth para la evaluación.
  - **Plotly, Matplotlib, WordCloud:** Para las visualizaciones de datos.
  - **Requests:** Para la adquisición de datos desde la API de Semantic Scholar.

## Instalación y Configuración

Sigue estos pasos para ejecutar el proyecto en tu máquina local.

**1. Prerrequisitos:**
   - Tener instalado Python 3.9 o superior.
   - Tener instalado `git`.

**2. Clonar el Repositorio**

**3. Crear y activar el entorno virtual:**
##### Crear el entorno virtual
python -m venv venv

##### Activar en Windows (PowerShell/CMD)
.\venv\Scripts\activate

##### Activar en MacOS/Linux
source venv/bin/activate

**4. Instalar dependencias:**
pip install -r requirements.txt

## Uso de la aplicacion:

**1. Ejecutar la aplicacion:**
desde la carpeta raiz del proyecto ejecuta esto en tu terminal:
streamlit run Recomendador.py

**2. Acceder a la aplicacion:**
Streamlit mostrará una URL local (normalmente http://localhost:8501) en la terminal. Abre esa URL en tu navegador web. Serás recibido por la pantalla de login. Ingresa las siguientes credenciales: USERNAME = "userTFM"  
PASSWORD = "password123"

