import os
import sys
from os import listdir, makedirs

def renombrar_imagenes(ruta_carpeta, inicio_numero):
    # Obtener la lista de archivos en la carpeta
    lista_archivos = os.listdir(ruta_carpeta)

    # Filtrar solo archivos con extensiones de imagen (puedes personalizar esto)
    archivos_imagen = filesInDir = [f for f in listdir(localpath) if f.endswith(".JPG")]

    # Ordenar archivos para asegurarse de que estén en el orden correcto
    archivos_imagen.sort()

    # Inicializar el contador
    numero = inicio_numero

    # Iterar sobre archivos y renombrar
    for archivo in archivos_imagen:
        # Construir el nuevo nombre de archivo

        nuevo = str(numero) + ".jpg"
        # Ruta completa del archivo antiguo y nuevo
        ruta_antiguo = os.path.join(ruta_carpeta, archivo)
        ruta_nuevo = os.path.join(ruta_carpeta, nuevo)

        # Renombrar el archivo
        os.rename(ruta_antiguo, ruta_nuevo)

        # Incrementar el contador
        numero += 1

if __name__ == "__main__":
    # Verificar si se proporcionó el número de inicio como argumento
    # if len(sys.argv) != 2:
    #     print("Uso: python script.py <numero_inicio>")
    #     sys.exit(1)

    # Configurar la carpeta raíz y parámetros de renombrado
    carpeta_raiz = localpath = os.getcwd();
    numero_inicio = int(sys.argv[1])
    # Llamar a la función para renombrar imágenes
    renombrar_imagenes(carpeta_raiz, numero_inicio)

    print("Renombrado completado.")
