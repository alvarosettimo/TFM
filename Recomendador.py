# front con streamlit

import streamlit as st
import sys
import os
import pandas as pd 
import time
from PIL import Image
from src.funciones_auxiliares import get_image_as_base64

# Obtener la ruta del directorio actual (donde esta recomender.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Añadir la ruta al directorio 'src' al sys.path
src_path = os.path.join(current_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

# importar desde src
try:
    from motor_de_recomendaciones import get_recommendations, get_similar_to_paper
except ImportError as e:
    st.error(f"Error al importar el motor de recomendación: {e}")
    st.stop() # Detiene la ejecución de la app si no se puede importar

# --- 2. Configuración de la Página de Streamlit ---
logo_img_pil = Image.open("C:\\Users\\alvar\\OneDrive\\Escritorio\\TFM\\static\\logo_app2.png")
st.set_page_config(
    page_title="Recomendador Bibliográfico",
    page_icon=logo_img_pil,
    layout="wide", # 'centered' o 'wide'
    initial_sidebar_state="expanded" # 'auto', 'expanded', 'collapsed'
)

# --- Inicializar st.session_state ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'login_attempt_username' not in st.session_state: # Para recordar el username entre intentos
    st.session_state.login_attempt_username = ""
if 'login_error_message' not in st.session_state:
    st.session_state.login_error_message = ""

if 'recommendations_to_display' not in st.session_state:
    st.session_state.recommendations_to_display = []
if 'display_query_context' not in st.session_state:
    st.session_state.display_query_context = ""
if 'current_query_text' not in st.session_state:
    st.session_state.current_query_text = ""

# --- CREDENCIALES ---
VALID_USERNAME = "userTFM"  
VALID_PASSWORD = "password123" 


# --- LÓGICA DE LOGIN Y APLICACIÓN PRINCIPAL ---
if not st.session_state.authenticated:
    login_logo_cols = st.columns([1, 1, 1]) # Para centrar el logo
    with login_logo_cols[1]:
         st.image("C:\\Users\\alvar\\OneDrive\\Escritorio\\TFM\\static\\logo_app2.png", width=150) 
    login_col_spacer1, login_col_form, login_col_spacer2 = st.columns([0.5,2.2, 1]) 

    with login_col_form:
        #st.image("C:\\Users\\alvar\\OneDrive\\Escritorio\\TFM\\static\\logo_app_recomendador.jpg", width=100) # Logo de la app
        st.markdown("Por favor, ingresa tus credenciales para acceder al recomendador bibliografico.")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Usuario", value=st.session_state.login_attempt_username, key="login_username")
            password = st.text_input("Contraseña", type="password", key="login_password")
            login_button = st.form_submit_button("Ingresar")

            if login_button:
                st.session_state.login_attempt_username = username
                if username == VALID_USERNAME and password == VALID_PASSWORD:
                    st.session_state.authenticated = True
                    st.session_state.login_error_message = ""
                    st.session_state.login_attempt_username = ""
                    st.rerun()
                else:
                    st.session_state.login_error_message = "Usuario o contraseña incorrectos."
            
        if st.session_state.login_error_message: # Mostrar error si existe
            st.error(st.session_state.login_error_message)
else:
    # --- 3. Interfaz de Usuario --- 
    st.title("Recomendador de Recursos Bibliográficos")
    st.markdown("""
    Bienvenido a tu asistente de investigación. Ingresa tu objetivo de estudio o área de interés
    y te sugeriremos artículos científicos, libros y otros materiales relevantes.
    """)

    # --- Sidebar para Entradas del Usuario ---
    with st.sidebar:
        #st.image("C:\\Users\\alvar\\OneDrive\\Escritorio\\TFM\\static\\logo_app.png", width=150)
        # --- centrado del logo ---
        logo_path = "C:\\Users\\alvar\\OneDrive\\Escritorio\\TFM\\static\\logo_app2.png"  
        logo_base64_encoded = get_image_as_base64(logo_path)
        st.markdown(
                f"""
                <div style="display: flex; justify-content: center; margin-bottom: 0px;">
                    <img src="data:image/png;base64,{logo_base64_encoded}" alt="Logo App" width="100" style="position: relative; left: -15px;">
                </div>
                """,
                unsafe_allow_html=True)
        
        st.header("🎯 Tu Objetivo de Estudio")
        user_query_input = st.text_area( 
            "Describe aquí tu tema de interés:",
            height=150,
            placeholder="Ej: Bibliographic resource recommender system using NLP"
        )
        
        top_n_input = st.number_input(
            "Número de recomendaciones a mostrar:",
            min_value=1, max_value=10, value=3, step=1
        )

        if st.button("Buscar Recomendaciones", key="main_search_button"):
                if not user_query_input.strip():
                    st.warning("Por favor, ingresa un objetivo de estudio.")
                    # Limpiar estado si la consulta es vacía
                    st.session_state.recommendations_to_display = []
                    st.session_state.display_query_context = ""
                else:
                    # 1. fijar la consulta del usuario para mostrarla despues
                    st.session_state.display_query_context = f"Resultados para: \"{user_query_input[:70]}{'...' if len(user_query_input) > 70 else ''}\""
                    # 2. mostrar el spinner mientras se buscan recomendaciones
                    with st.spinner("Buscando recomendaciones y procesando tu consulta..."):
                        st.session_state.recommendations_to_display = get_recommendations(user_query=user_query_input, top_n=top_n_input)
                        time.sleep(2) # Simular tiempo de procesamiento para que se vea el mensaje del spinner
                st.rerun() # Importante!! para que el flujo de Streamlit actualice la UI con el nuevo estado

        if st.session_state.recommendations_to_display:
            if st.button("Limpiar Resultados", key="clear_results_button"):
                st.session_state.recommendations_to_display = []
                st.session_state.display_query_context = ""
                st.rerun()

    # --- Visualización de Resultados ---
    if st.session_state.display_query_context:
        st.markdown("---")
        st.subheader(st.session_state.display_query_context)

    if isinstance(st.session_state.recommendations_to_display, list):
        if st.session_state.recommendations_to_display:
            st.success(f"¡Se encontraron {len(st.session_state.recommendations_to_display)} recomendaciones!")
            
            for rec in st.session_state.recommendations_to_display:
                with st.container(): # Contenedor para cada recomendación
                    
                    # --- SECCIÓN PARA TÍTULO Y BOTÓN "MÁS COMO ESTE" ---
                    col_titulo, col_boton_similar = st.columns([4, 1]) # Ej: Título ocupa 4/5 del espacio, botón 1/5
                                                                    
                    
                    with col_titulo:
                        st.markdown(f"#### {rec.get('rank', '')}. {rec.get('title', 'Título no disponible')}")
                    
                    with col_boton_similar:
                        button_key_similar = f"similar_to_{rec.get('paperId', str(rec.get('rank')))}"
                        if st.button("Más como este 👍", key=button_key_similar, use_container_width=True): 
                            with st.spinner(f"Buscando artículos similares a '{rec.get('title', 'este paper')[:30]}...'"):
                                st.session_state.recommendations_to_display = get_similar_to_paper(seed_paper_id=rec['paperId'], top_n=top_n_input)
                                st.session_state.display_query_context = f"Similares a: \"{rec.get('title', 'Paper Seleccionado')[:70]}{'...' if len(rec.get('title', 'Paper Seleccionado')) > 70 else ''}\""
                                time.sleep(2) 
                            st.rerun()
                    # --- FIN SECCIÓN ---

                    # Columnas para Score, Año y Autores (debajo del título y botón "Más como este")
                    col_metricas1, col_metricas2, col_metricas3 = st.columns([1.5,0.5,2])
                    with col_metricas1:
                        st.metric(label="Score", value=f"{rec.get('similarity_score', 0):.4f}")
                    with col_metricas2:
                        st.info(f"**Año:** {rec.get('year', 'N/A')}")
                    with col_metricas3:
                        st.info(f"**Autores:** {rec.get('authors', 'N/A')}")

                    # Enlace PDF y Snippet del Abstract
                    pdf_url = rec.get('pdf_url')
                    if pdf_url and isinstance(pdf_url, str) and pdf_url.lower() != 'nan' and pdf_url.startswith('http'):
                        st.markdown(f"🔗 [Acceder al PDF]({pdf_url})", unsafe_allow_html=True)
                    else:
                        st.caption("PDF no disponible o URL inválida.")
                    
                    with st.expander("Ver abstract"):
                        st.markdown(f"*{rec.get('abstract', 'Abstract no disponible...')}*")
                    
                    # Botones de Feedback
                    st.caption("¿Te pareció útil esta recomendación?")
                    feedback_cols = st.columns(2)

                    with feedback_cols[0]:
                        button_key_relevant = f"relevant_{rec.get('paperId', str(rec.get('rank')))}"
                        if st.button("👍 Relevante", key=button_key_relevant, use_container_width=True):
                            st.toast(f"Feedback: '{rec.get('title', rec['paperId'])[:30]}...' marcado como RELEVANTE.", icon="👍")

                    with feedback_cols[1]:
                        button_key_not_relevant = f"not_relevant_{rec.get('paperId', str(rec.get('rank')))}"
                        if st.button("👎 No Relevante", key=button_key_not_relevant, use_container_width=True):
                            st.toast(f"Feedback: '{rec.get('title', rec['paperId'])[:30]}...' marcado como NO RELEVANTE.", icon="👎")
                    
                    st.markdown("---") # Separador entre recomendaciones

        elif st.session_state.display_query_context:
            st.info("No se encontraron recomendaciones que coincidan con tu criterio.")

    elif isinstance(st.session_state.recommendations_to_display, str): # Si es un string de error del backend
        st.error(st.session_state.recommendations_to_display)

        

    # --- Información Adicional o Pie de Página ---
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Este recomendador utiliza técnicas de NLP y Machine Learning "
        "para sugerir recursos basados en la similitud semántica."
    )