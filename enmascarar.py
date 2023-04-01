from transformers import pipeline
import re

# Texto de ejemplo
texto = """
El número de mi DNI es 12345678A y vivo en la calle Gran Vía de Madrid, en el barrio de Chueca.
"""

# Inicializar el modelo NER de Hugging Face
modelo_ner = pipeline("ner", model="NOMBRE_DEL_MODELO", tokenizer="NOMBRE_DEL_TOKENIZER")

# Diccionario de tipos de datos sensibles y sus placeholders correspondientes
tipos_sensibles = {
    "PER": "[NOMBRE]",
    "ORG": "[ORGANIZACIÓN]",
    "LOC": "[LUGAR]",
    "MISC": "[DATO]"
}

# Función para enmascarar los tipos de datos sensibles y números de DNI
def enmascarar_datos_sensibles(texto, modelo_ner, tipos_sensibles):
    # Detectar los tipos de datos sensibles en el texto
    entidades = modelo_ner(texto)
    tipos_detectados = set([entidad["entity_group"] for entidad in entidades])

    # Reemplazar los tipos de datos sensibles por los placeholders correspondientes
    for tipo in tipos_detectados:
        if tipo in tipos_sensibles:
            texto = texto.replace(tipo, tipos_sensibles[tipo])

    # Enmascarar números de DNI
    dnis = re.findall(r'\b(\d{8}[a-zA-Z]|\d{7}[a-zA-Z]{1,2})\b', texto)
    for dni in dnis:
        texto = texto.replace(dni, "[DNI]")

    return texto

# Enmascarar los tipos de datos sensibles y números de DNI en el texto de ejemplo
texto_enmascarado = enmascarar_datos_sensibles(texto, modelo_ner, tipos_sensibles)

# Imprimir el texto enmascarado
print(texto_enmascarado)
