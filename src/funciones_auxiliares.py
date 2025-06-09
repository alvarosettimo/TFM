# --- FUNCIONES AUXILIARES ---
import base64



# --- FUNCION PARA CENTRAR EL LOGO ---
def get_image_as_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()