from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

# Cargamos las variables de entorno
load_dotenv()

# Funcion principal para probar la conexion
def probar_conexion_modelo():
    print("Iniciando prueba de conexion con el modelo de lenguaje...")

    try:
        # Intentamos inicializar el modelo.
        load_dotenv()

        #nombre_modelo = "qwen/qwen3.8-27b"
        nombre_modelo = "modelo/que-no-existe-12345"
        modelo = ChatGroq(
            model=nombre_modelo,
            temperature=0.0
)

        print("Modelo inicializado en memoria. Enviando peticion...")

        mensaje = HumanMessage(content="Hola, dime un chiste corto.")

        # Esta es la linea que realmente dispara la peticion HTTP al proveedor.
        respuesta = modelo.invoke([mensaje])

        print("Exito:", respuesta.content)

    except Exception as error_general:
        # Captura generica de seguridad
        print("Ocurrio un error inesperado:", str(error_general))

# Ejecucion del script
if __name__ == "__main__":
    probar_conexion_modelo()

'''
from groq import AuthenticationError, NotFoundError, APIConnectionError

try:
    ...
except AuthenticationError as e:
    print("Error de autenticación. Revisa tu GROQ_API_KEY.", e)
except NotFoundError as e:
    print("Modelo no encontrado. Revisa el nombre.", e)
except APIConnectionError as e:
    print("Sin conexión con el servidor de Groq.", e)
except Exception as e:
    print("Error inesperado:", e)
'''