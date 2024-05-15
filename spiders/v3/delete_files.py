import os
import shutil

def eliminar_archivos_por_nombre(dir_origen, dir_destino):
    # Obtener la lista de nombres de archivos en el directorio de destino
    nombres_archivos_destino = set(os.path.splitext(archivo)[0] for archivo in os.listdir(dir_destino))

    # Iterar sobre los archivos en el directorio de origen
    for archivo_origen in os.listdir(dir_origen):
        nombre_archivo_origen, extension = os.path.splitext(archivo_origen)

        # Verificar si el nombre del archivo no está presente en el directorio de destino
        if nombre_archivo_origen not in nombres_archivos_destino:
            try:
                # Eliminar el archivo
                ruta_origen = os.path.join(dir_origen, archivo_origen)
                if os.path.isfile(ruta_origen):
                    os.remove(ruta_origen)
                    print(f"Archivo eliminado: {archivo_origen}")
                    
            except Exception as e:
                print(f"No se pudo eliminar el archivo {archivo_origen}: {e}")

# Directorios de ejemplo (reemplázalos con tus directorios reales)
directorio_origen = "./target"
directorio_destino = "./input"

eliminar_archivos_por_nombre(directorio_origen, directorio_destino)
