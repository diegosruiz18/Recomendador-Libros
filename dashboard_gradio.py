#En este archivo creamos la interfaz de usuario para interactuar
#con el recomendador de libros.

#Importando librerías
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import os
import zipfile

#Langchain
from langchain_community.document_loaders import TextLoader #cargador de texto
from langchain_text_splitters import CharacterTextSplitter #divisor de texto en fragmentos significativos
from langchain_openai import OpenAIEmbeddings #trabaja con los fragmentos (usamos OpenAI)
from langchain_chroma import Chroma #base de datos vectorial
#from langchain.vectorstores import Chroma #versión de Chroma con persistencia

#Gradio
import gradio as gr

#Carga de variables del archivo .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Carga de datos
libros = pd.read_csv("books_with_emotions_translated.csv")

#Obtener imágenes de mayor tamaño (portada de libros)
libros["large_thumbnail"] = libros["thumbnail"] + "&fife=w800"

#Si no hay imagen de portada se le asigna un fondo que indica vacío
libros["large_thumbnail"] = np.where(
    libros["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    libros["large_thumbnail"]
)

#Crear base de datos vectorial
documentos_sin_procesar = TextLoader("tagged_description.txt", encoding="utf-8").load()
divisor_texto = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documentos = divisor_texto.split_documents(documentos_sin_procesar)

#Base de datos vectorial Chroma
"""
#Primera ejecución (persistencia db_libros)
db_libros = Chroma.from_documents(
    documentos,
    embedding=OpenAIEmbeddings(), #modelo_incrustacion
    #persistencia
    persist_directory="db_libros"
)
db_libros.persist()
"""
#A partir de la segunda ejecución
db_libros = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="db_libros"
)

#Función para traducir de español a inglés
def traducir_espanol_a_ingles(texto):
    respuesta = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Traduce el siguiente texto del español al inglés, manteniendo el significado exacto."},
            {"role": "user", "content": texto}
        ],
        temperature=0,
        max_tokens=1000
    )
    return respuesta.choices[0].message.content.strip()

#Función que recupera recomendaciones, aplica filtros basados en categorías y ordena por emociones
def recuperar_recomendaciones_semanticas(
        consulta: str,
        categoria: str = None,
        tono: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:
    respuesta = db_libros.similarity_search(query=consulta, k=initial_top_k) #obtener recomendaciones de bd vectorial
    lista_libros = [int(rpta.page_content.strip('"').split()[0]) for rpta in respuesta] #obtener codigo isbn13
    respuesta_libros = libros[libros["isbn13"].isin(lista_libros)].head(final_top_k) #guardando libros recomendados

    #Aplicar filtros basados en categoría
    if categoria != "Todos":
        #Se filtra según categoría elegida
        respuesta_libros = respuesta_libros[respuesta_libros["simple_categories"] == categoria].head(final_top_k)
    else:
        #Devuelve todas las recomendaciones
        respuesta_libros = respuesta_libros.head(final_top_k)

    #Ordenando según probabilidad de emoción
    if tono == "Felicidad":
        respuesta_libros.sort_values(by="joy", ascending=False, inplace=True)
    elif tono == "Sorpresa":
        respuesta_libros.sort_values(by="surprise", ascending=False, inplace=True)
    elif tono == "Enojo":
        respuesta_libros.sort_values(by="anger", ascending=False, inplace=True)
    elif tono == "Suspenso":
        respuesta_libros.sort_values(by="fear", ascending=False, inplace=True)
    elif tono == "Tristeza":
        respuesta_libros.sort_values(by="sadness", ascending=False, inplace=True)

    #Retornando recomendaciones
    return respuesta_libros

#Función que especifica lo que se mostrará en el dashboard
def recomendar_libros(consulta: str, categoria: str, tono: str):
    #Validar consulta nula o vacía
    if not consulta or str(consulta).strip() == "":
        return []

    #Traducir consulta del español a inglés para recomendaciones
    consulta = traducir_espanol_a_ingles(consulta)

    #Realizando búsqueda
    recomendaciones = recuperar_recomendaciones_semanticas(consulta, categoria, tono)
    resultados = []

    #Recorriendo recomendaciones
    for _, row in recomendaciones.iterrows():
        #Descripciones
        descripcion = row["spanish_description"] #descripción traducida al español
        descripcion_dividida = descripcion.split()
        #Dividiendo las descripciones en palabras separadas, si contiene más de 30 palabras se corta
        descripcion_truncada = " ".join(descripcion_dividida[:30]) + "..."

        #Autores
        autores_dividido = row["authors"].split(";")
        if len(autores_dividido) == 2: #si el libro tiene 2 autores
            autores_str = f"{autores_dividido[0]} y {autores_dividido[1]}" #and
        elif len(autores_dividido) > 2: #si el libro tiene múltiples autores
            autores_str = f"{', '.join(autores_dividido[:-1])}, y {autores_dividido[-1]}" #and
        else:
            autores_str = row["authors"]

        #Combinando lo anterior
        subtitulo_informacion = f"{row['title']} por {autores_str}: {descripcion_truncada}" #by

        #Asignar miniatura y subtitulo en una tupla y guardar en lista
        resultados.append((row["large_thumbnail"], subtitulo_informacion))
    return resultados #se retorna la lista

#Crear el dashboard
categorias = ["Todos"] + sorted(libros["simple_categories"].unique())
tonos = ["Todos"] + ["Felicidad", "Sorpresa", "Enojo", "Suspenso", "Tristeza"]

#Tema del dashboard
with gr.Blocks(theme=gr.themes.Ocean()) as dashboard:
    #Configuración de estilos
    gr.HTML("""
    <style>
        caption.caption.svelte-1atirkn {
            font-size: 12px !important;         /* Tamaño más pequeño */
            line-height: 1.4;
            white-space: normal !important;
            text-align: justify !important;
            padding: 0.5em;
            max-width: 100%;
            overflow-wrap: break-word;
        }
    </style>
    """)

    #Contenido
    gr.Markdown("# Recomendador de libros semántico")

    with gr.Row():
        #Input
        consulta_usuario = gr.Textbox(label="Porfavor ingrese una descripción:",
                                placeholder="Por ejemplo, una historia acerca del perdón.")
        #Menú despegable
        categoria_dropdown = gr.Dropdown(choices=categorias, label="Seleccione categoría:", value="Todos",
                                         allow_custom_value=False)
        tono_dropdown = gr.Dropdown(choices=tonos, label="Seleccione tono emocional:", value="Todos",
                                    allow_custom_value=False)
        #Botón
        submit_button = gr.Button("Buscar libros")

    gr.Markdown("## Recomendaciones")
    output = gr.Gallery(label="Libros recomendados", columns=8, rows=2)

    #Al hacer click en el botón se ejecuta lo siguiente
    submit_button.click(fn=recomendar_libros,
                        inputs=[consulta_usuario, categoria_dropdown, tono_dropdown],
                        outputs=output)

#Método main
if __name__ == "__main__":
    dashboard.launch() #ejecutar dashboard