from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import json

def cargar_plantilla_json(ruta: str) -> ChatPromptTemplate:
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    mensajes = [("system", datos["system_message"])]

    for ejemplo in datos["few_shot_examples"]:
        mensajes.append(("human", ejemplo["human"]))
        mensajes.append(("ai", ejemplo["ai"]))

    mensajes.append(("human", datos["input_template"]))
    return ChatPromptTemplate.from_messages(mensajes)                 

load_dotenv()

""" plantilla_few_shot = ChatPromptTemplate.from_messages(
    [
        ("system", "Eres un extractor de entidades. Devuelve siempre la respuesta en formato JSON válido."),

        # Ejemplo 1 - LLAVES ESCAPADAS con {{ }}
        ("human", "Texto: El iPhone 15 Pro cuesta 1199 dólares y tiene un chip A17 Pro."),
        ("ai", '{{"producto": "iPhone 15 Pro", "precio": "1199 dólares", "chip": "A17 Pro"}}'),

        # Ejemplo 2 - LLAVES ESCAPADAS con {{ }}
        ("human", "Texto: La tablet Galaxy Tab S9 viene con 256GB de almacenamiento por 850 euros."),
        ("ai", '{{"producto": "Galaxy Tab S9", "precio": "850 euros", "almacenamiento": "256GB"}}'),

        # Entrada real del usuario - Esta llave SÍ es una variable
        ("human", "Texto: {texto_usuario}"),
    ]
) """

plantilla = cargar_plantilla_json("./leccion_2/extractor_entidades.json")

texto_a_procesar = "La cámara Sony A7 IV tiene un sensor de 33 megapíxeles y cuesta 2500 dólares."
mensajes_con_ejemplos = plantilla.format_messages(texto_usuario=texto_a_procesar)

modelo = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.0)
respuesta_estructurada = modelo.invoke(mensajes_con_ejemplos)
print("\nRespuesta:\n", respuesta_estructurada.content)