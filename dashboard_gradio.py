#En este archivo creamos la interfaz de usuario para interactuar
#con el recomendador de libros.

#In this file the user interface is created to interact with
#book recommender system, then the application will be deployed

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
libros = pd.read_csv("data/books_with_emotions_translated.csv")

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
    if categoria != "All":
        #Se filtra según categoría elegida
        respuesta_libros = respuesta_libros[respuesta_libros["simple_categories"] == categoria].head(final_top_k)
    else:
        #Devuelve todas las recomendaciones
        respuesta_libros = respuesta_libros.head(final_top_k)

    #Ordenando según probabilidad de emoción
    if tono == "Happy":
        respuesta_libros.sort_values(by="joy", ascending=False, inplace=True)
    elif tono == "Surprising":
        respuesta_libros.sort_values(by="surprise", ascending=False, inplace=True)
    elif tono == "Angry":
        respuesta_libros.sort_values(by="anger", ascending=False, inplace=True)
    elif tono == "Suspenseful":
        respuesta_libros.sort_values(by="fear", ascending=False, inplace=True)
    elif tono == "Sad":
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
        descripcion = row["description"]
        descripcion_dividida = descripcion.split()
        #Dividiendo las descripciones en palabras separadas, si contiene más de 30 palabras se corta
        descripcion_truncada = " ".join(descripcion_dividida[:30]) + "..."

        #Autores
        autores_dividido = row["authors"].split(";")
        if len(autores_dividido) == 2: #si el libro tiene 2 autores
            autores_str = f"{autores_dividido[0]} and {autores_dividido[1]}"
        elif len(autores_dividido) > 2: #si el libro tiene múltiples autores
            autores_str = f"{', '.join(autores_dividido[:-1])}, and {autores_dividido[-1]}"
        else:
            autores_str = row["authors"]

        #Combinando lo anterior
        subtitulo_informacion = f"{row['title']} by {autores_str}: {descripcion_truncada}"

        #Asignar miniatura y subtitulo en una tupla y guardar en lista
        resultados.append((row["large_thumbnail"], subtitulo_informacion))
    return resultados #se retorna la lista

#Crear el dashboard
categorias = ["All"] + sorted(libros["simple_categories"].unique())
tonos = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

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
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        #Input
        consulta_usuario = gr.Textbox(label="Please enter a description:", #Porfavor ingrese una descripción
                                placeholder="E.g. a story about World War II") #Por ejemplo, una historia acerca de la segunda guerra mundial
        #Menú despegable
        categoria_dropdown = gr.Dropdown(choices=categorias, label="Select category:", value="All", #Seleccione categoría
                                         allow_custom_value=False)
        tono_dropdown = gr.Dropdown(choices=tonos, label="Select emotional tone:", value="All", #Seleccione tono emocional
                                    allow_custom_value=False)
        #Botón
        submit_button = gr.Button("Find books") #Buscar libros

    gr.Markdown("## Recomendations")
    output = gr.Gallery(label="Books recommended", columns=8, rows=2)

    #Al hacer click en el botón se ejecuta lo siguiente
    submit_button.click(fn=recomendar_libros,
                        inputs=[consulta_usuario, categoria_dropdown, tono_dropdown],
                        outputs=output)

#Método main
if __name__ == "__main__":
    dashboard.launch() #ejecutar dashboard