"""
import pandas as pd
import numpy as np
import re
import os
#import pruebaKNN as knn

def normalizar_csv(data, archivo_salida):
    #Normaliza los valores de todas las columnas numéricas de un DataFrame al rango [0, 1].
    try:
        # Identificar las columnas numéricas
        columnas_numericas = data.select_dtypes(include=['float64', 'int64']).columns
        # Normalizar las columnas numéricas al rango [0, 1]
        data[columnas_numericas] = data[columnas_numericas].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        # Crear carpeta si no existe
        os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
        # Guardar el DataFrame normalizado
        data.to_csv(archivo_salida, index=False)
    except Exception as e:
        print(f"Error al normalizar y guardar el archivo CSV: {e}")

def funcionPrincipalCalculoArchivo(base_directory):
    print('empezar el procesamiento', base_directory)
    def orden_natural(nombre_archivo):
        partes = re.split(r'(\d+)', nombre_archivo)
        partes = [int(parte) if parte.isdigit() else parte for parte in partes]
        return partes

    def procesar_archivo_txt(archivo_txt_path):
        with open(archivo_txt_path, "r") as archivo:
            lineas = archivo.readlines()
        coordenadas_por_persona = {}
        persona_actual = None
        for linea in lineas:
            if "Person" in linea:
                persona_actual = int(linea.strip().split()[-1].replace(':', ''))
                coordenadas_por_persona[persona_actual] = []
            elif persona_actual is not None:
                valores = [float(s) for s in linea.replace(",", " ").split()]
                if len(valores) >= 2:
                    coordenadas_por_persona[persona_actual].append((valores[0], valores[1]))
        return coordenadas_por_persona

    def desviacionEstandar(coordenadas):
        try:
            coordenadas_np = np.array(coordenadas)
            if coordenadas_np.ndim == 2:
                media_x = np.mean(coordenadas_np[:, 0])
                media_y = np.mean(coordenadas_np[:, 1])
                distancias = np.sqrt((coordenadas_np[:, 0] - media_x)**2 + (coordenadas_np[:, 1] - media_y)**2)
            else:
                media_x = np.mean(coordenadas_np)
                distancias = np.sqrt((coordenadas_np - media_x)**2)
            desviacion_estandar = np.std(distancias)
            return desviacion_estandar
        except ZeroDivisionError:
            return 0
    
    def promedio(coordenadas):
        distancias_al_origen = [np.sqrt(x**2 + y**2) for x, y in coordenadas]
        if len(distancias_al_origen) != 0:
            promedio_distancias = sum(distancias_al_origen) / len(distancias_al_origen)
        else:
            promedio_distancias = 0
        return promedio_distancias
    
    def calcular_simetria(cantidadMov):
        #Calcula la simetría S_J entre ciclos de movimiento en los ejes X e Y.
        simetria_results = []
        for i in range(len(cantidadMov) - 1):
            # Obtener desviaciones estándar para el ciclo 1 y ciclo 2
            desv_actual = cantidadMov[i][5]  # Desviación estándar del ciclo actual
            desv_siguiente = cantidadMov[i + 1][6]  # Desviación estándar del ciclo siguiente
            # Calcular S_J para X y Y
            if desv_siguiente != 0:
                S_J = desv_actual / desv_siguiente
            else:
                S_J = 0  # Evitar división por cero
            # Determinar si cumple con el criterio
            QoM_izqJ = cantidadMov[i][2]  # Asumimos que QoM_izqJ corresponde a cantidad_movimiento
            cumple_criterio = S_J >= QoM_izqJ
            # Agregar a los resultados
            simetria_results.append(cantidadMov[i] + [S_J, cumple_criterio])
        return simetria_results
    
    def procesar_pares_de_archivos(archivos_txt, archivo_salida):
        print('empezar a procesar archivos')
        cantidadMov = []
        try:
            for i in range(len(archivos_txt) - 1):
                archivo_actual_path = archivos_txt[i]
                archivo_siguiente_path = archivos_txt[i + 1]
                coordenadas_imagen_actual = procesar_archivo_txt(archivo_actual_path)
                coordenadas_imagen_siguiente = procesar_archivo_txt(archivo_siguiente_path)
                for persona_actual in coordenadas_imagen_actual.keys():
                    coords_actual = coordenadas_imagen_actual[persona_actual]
                    for persona_siguiente in coordenadas_imagen_siguiente.keys():
                        coords_siguiente = coordenadas_imagen_siguiente[persona_siguiente]
                        diferencia = [((siguiente[0] - actual[0])**2, (siguiente[1] - actual[1])**2) 
                                    for actual, siguiente in zip(coords_actual, coords_siguiente)]
                        raiz_cuadrada_diferencia = [np.sqrt(dif[0] + dif[1]) for dif in diferencia]
                        cantidad_movimiento = np.sum(raiz_cuadrada_diferencia)
                        promedioImagenActual = promedio(coords_actual)
                        promedioImagenSiguiente = promedio(coords_siguiente)
                        desviacionActual = desviacionEstandar(coords_actual)
                        desviacionSiguiente = desviacionEstandar(coords_siguiente)
                        if promedioImagenActual != 0:
                            homogeneidad_Actual = (promedioImagenActual - desviacionActual) / promedioImagenActual
                        else:
                            homogeneidad_Actual = 0
                        if promedioImagenSiguiente != 0:
                            homogeneidad_Siguiente = (promedioImagenSiguiente - desviacionSiguiente) / promedioImagenSiguiente
                        else:
                            homogeneidad_Siguiente = 0
                        cantidadMov.append([archivo_actual_path, archivo_siguiente_path,
                                            cantidad_movimiento, 
                                            promedioImagenActual, promedioImagenSiguiente,
                                            desviacionActual, desviacionSiguiente, 
                                            homogeneidad_Actual, homogeneidad_Siguiente])
        except Exception as e:
            print(f"Error al procesar pares de archivos: {e}")
        # Calcular simetría y normalizar resultados
        simetria_results = calcular_simetria(cantidadMov)
        # Crear DataFrame y normalizar
        columnas = ['Archivo Actual', 'Archivo Siguiente',
                    'Cantidad Movimiento', 'Promedio Actual', 
                    'Promedio Siguiente', 'Desviacion Actual', 'Desviacion Siguiente', 
                    'Homogeneidad Actual', 'Homogeneidad Siguiente', 
                    'Simetria S_J', 'Cumple Criterio']
        df = pd.DataFrame(simetria_results, columns=columnas)
        normalizar_csv(df, archivo_salida)
    txt_base_folder = os.path.join(base_directory, 'txt')
    print('txt_base_folder', txt_base_folder)
    cantidad_mov_base_folder = os.path.join(base_directory, 'cantidadMovimiento')
    print('cantidad_mov_base_folder', cantidad_mov_base_folder)
    for folder_name in os.listdir(txt_base_folder):
        folder_path = os.path.join(txt_base_folder, folder_name)
        if os.path.isdir(folder_path):
            # Procesar subcarpetas dentro de cada carpeta
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
                if os.path.isdir(subfolder_path):
                    cantidad_mov_folder = os.path.join(cantidad_mov_base_folder, folder_name)
                    archivo_salida = os.path.join(cantidad_mov_folder, f'{cantidad_mov_folder +"_"+ subfolder_name}.csv')
                    print('Procesando subcarpeta:', subfolder_path)
                    print('Archivo de salida:', archivo_salida)
                    archivos_txt = sorted([os.path.join(subfolder_path, archivo) 
                                           for archivo in os.listdir(subfolder_path) 
                                           if archivo.endswith('.txt')],
                                          key=orden_natural)
                    if archivos_txt:
                        procesar_pares_de_archivos(archivos_txt, archivo_salida)
                    else:
                        print(f"No hay archivos .txt en la subcarpeta {subfolder_path}")
    print("Procesamiento completado.")

if __name__ == "__main__":
    #base_directory = "D:/programs/redNeuronal/modificaciones/datos agrupar"
    base_directory ="D:/programs/redNeuronal/modificaciopnes/programa/datos"
    funcionPrincipalCalculoArchivo(base_directory)
"""



from pathlib import Path
import pandas as pd
import numpy as np
import re
import os

def normalizar_csv(data, archivo_salida):
    try:
        columnas_numericas = data.select_dtypes(include=['float64', 'int64']).columns
        data[columnas_numericas] = data[columnas_numericas].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
        data.to_csv(archivo_salida, index=False)
    except Exception as e:
        print(f"Error al normalizar y guardar el archivo CSV: {e}")
        
def funcionPrincipalCalculoArchivo(txt_base_folder, base_directory):    
        
    def orden_natural(nombre_archivo):
        partes = re.split(r'(\d+)', nombre_archivo)
        return [int(p) if p.isdigit() else p.lower() for p in partes if p]

    def procesar_archivo_txt(archivo_txt_path):
        with open(archivo_txt_path, "r") as archivo:
            lineas = archivo.readlines()    
        coordenadas_por_persona = {}
        persona_actual = None    
        for linea in lineas:
            linea = linea.strip()
            if "Person" in linea:
                try:
                    # Extraer número de persona (maneja "Person :" o "Person")
                    partes = linea.replace(':', '').split()
                    if len(partes) >= 1 and partes[-1].isdigit():
                        persona_actual = int(partes[-1])
                    else:
                        # Asignar un ID único si no hay número (ej: -1)
                        persona_actual = -1  
                    coordenadas_por_persona[persona_actual] = []
                except:
                    persona_actual = -1
                    coordenadas_por_persona[persona_actual] = []        
            elif persona_actual is not None:
                # Procesar coordenadas incluso si son (0.0, 0.0)
                valores = [float(s) for s in linea.replace(",", " ").split()]
                if len(valores) >= 2:
                    coordenadas_por_persona[persona_actual].append((valores[0], valores[1]))     
        return coordenadas_por_persona

    def desviacionEstandar(coordenadas):
        if not coordenadas:
            return 0.0
        try:
            coordenadas_np = np.array(coordenadas)
            if coordenadas_np.ndim == 2:
                media_x, media_y = np.mean(coordenadas_np, axis=0)
                distancias = np.sqrt((coordenadas_np[:,0] - media_x)**2 + (coordenadas_np[:,1] - media_y)**2)
            else:
                distancias = np.abs(coordenadas_np - np.mean(coordenadas_np))
            return np.std(distancias)
        except:
            return 0.0

    def promedio(coordenadas):
        if not coordenadas:
            return 0.0
        distancias = [np.sqrt(x**2 + y**2) for x, y in coordenadas]
        return np.mean(distancias) if distancias else 0.0

    def calcular_simetria(cantidadMov):
        simetria_results = []
        for i in range(len(cantidadMov)):  # Itera todos los pares
            desv_actual = cantidadMov[i][5]    # Desviación del patrón 1 (actual)
            desv_siguiente = cantidadMov[i][6] # Desviación del patrón 2 (siguiente)
            S_J = desv_actual / desv_siguiente if desv_siguiente != 0 else 0
            QoM_izqJ = cantidadMov[i][2]      # Criterio QoM
            simetria_results.append(cantidadMov[i] + [S_J, S_J >= QoM_izqJ])
        return simetria_results
    
    def procesar_pares_de_archivos(archivos_txt, archivo_salida):
        cantidadMov = []
        TARGET_FPS = 10
        try:
            for i in range(len(archivos_txt) - 1):
                archivo_actual = archivos_txt[i]
                archivo_siguiente = archivos_txt[i + 1]
                
                coords_actual = procesar_archivo_txt(archivo_actual)
                coords_siguiente = procesar_archivo_txt(archivo_siguiente)

                fps = 30  # Reemplazar con tu valor real
                frame_interval = max(1, int(fps // TARGET_FPS))
                delta_t = frame_interval / fps  # Tiempo real entre frames procesados
                
                # Procesar todas las combinaciones de personas
                for p_act in coords_actual:
                    if p_act in coords_siguiente:  # ✅ ¿Existe la misma persona en el siguiente archivo?
                        p_sig = p_act 
                        if not coords_actual[p_act] or not coords_siguiente[p_sig]:
                            cantidadMov.append([archivo_actual, archivo_siguiente, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                            continue                    
                        # Cálculo de movimiento
                        dif_x = np.array([s[0] - a[0] for a, s in zip(coords_actual[p_act], coords_siguiente[p_sig])])**2
                        dif_y = np.array([s[1] - a[1] for a, s in zip(coords_actual[p_act], coords_siguiente[p_sig])])**2
                        raices = np.sqrt(dif_x + dif_y)

                        # --- Cálculo de velocidad (nuevo) ---
                        velocidad = raices / delta_t  # Velocidad por keypoint
                        velocidad_promedio = np.mean(velocidad)  # Métrica agregada
                        
                        # Métricas
                        mov_total = np.sum(raices)
                        prom_act = promedio(coords_actual[p_act])
                        prom_sig = promedio(coords_siguiente[p_sig])
                        dev_act = desviacionEstandar(coords_actual[p_act])
                        dev_sig = desviacionEstandar(coords_siguiente[p_sig])
                        hom_act = (prom_act - dev_act)/prom_act if prom_act !=0 else 0.0
                        hom_sig = (prom_sig - dev_sig)/prom_sig if prom_sig !=0 else 0.0
                        
                        cantidadMov.append([
                            archivo_actual, archivo_siguiente,
                            mov_total, prom_act, prom_sig,
                            dev_act, dev_sig, hom_act, hom_sig,
                            velocidad_promedio])
                    else:
                        cantidadMov.append([archivo_actual, archivo_siguiente, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    continue
                   
            # Calcular simetría y guardar
            simetria_results = calcular_simetria(cantidadMov)
            df = pd.DataFrame(simetria_results, columns=[
                'Archivo Actual', 'Archivo Siguiente', 'Cantidad Movimiento',
                'Promedio Actual', 'Promedio Siguiente', 'Desviacion Actual',
                'Desviacion Siguiente', 'Homogeneidad Actual', 'Homogeneidad Siguiente',
                'Velocidad Promedio', 
                'Simetria S_J', 'Cumple Criterio'  ])
            os.makedirs(os.path.dirname(archivo_salida), exist_ok=True)
            normalizar_csv(df, archivo_salida)
            print(f"\n✅ Archivo consolidado guardado en: {archivo_salida}")                
            return df 
        except Exception as e:
            print(f"Error en procesar_pares_de_archivos: {str(e)}")
            return pd.DataFrame()
    
    # 1. Definir rutas críticas
    cantidad_mov_base_folder = os.path.join(base_directory, 'cantidadMovimiento')
    print(f"[DEBUG] base_directory: {base_directory}")
    print(f"[DEBUG] cantidad_mov_base_folder: {cantidad_mov_base_folder}")

    # 2. Crear carpeta "cantidadMovimiento"
    os.makedirs(cantidad_mov_base_folder, exist_ok=True)
    folder_name = os.path.basename(os.path.dirname(txt_base_folder))
    print(f"[DEBUG] folder_name: {folder_name}")

    # 4. Definir archivo de salida .csv
    archivo_salida = os.path.join(cantidad_mov_base_folder, f'{folder_name}.csv')
    print(f"[DEBUG] archivo_salida: {archivo_salida}")

    # 5. Procesar archivos .txt
    if os.path.isdir(txt_base_folder):
        archivos_txt = sorted(
            [os.path.join(txt_base_folder, archivo) for archivo in os.listdir(txt_base_folder) if archivo.endswith('.txt')],
                key=orden_natural)
        if archivos_txt:
                print(f"[DEBUG] Procesando {len(archivos_txt)} archivos .txt")
                df = procesar_pares_de_archivos(archivos_txt, archivo_salida)
                print("[DEBUG] Llamando a predecir...")
                import predecir_archivo as predecir
                predecir.funcionPrincipal(archivo_salida, base_directory)
        else:
            print(f"[ERROR] No hay archivos .txt en {txt_base_folder}")
    else:
        print(f"[ERROR] La ruta {txt_base_folder} no es un directorio válido")

                  

if __name__ == "__main__":    
    #funcionPrincipalCalculoArchivo()    
    #txt_base_folder="D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet/video_04-04/video_20250303_163800/txt"
    #txt_base_folder="D:/programs/redNeuronal/modificaciones/datos_agrupar\dataSet/video_04-04/segmentos Persona 3 video_20240214_133653_2/etiquetas"
    
    #txt_base_folder="D:/programs/redNeuronal/modificaciones/programa_final/datos/segmentos Persona 3 video_20240214_133653_4/txt"
    #base_directory="D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet/video_04-04/dos personas"
    #base_directory="D:/programs/redNeuronal/modificaciones/programa_final/datos"
    
    import time
    start_time = time.time()  # Iniciar temporizador
    #funcionPrincipalCalculoArchivo(txt_base_folder, base_directory)
    funcionPrincipalCalculoArchivo() 
    end_time = time.time()  # Fin del temporizador
    duration = end_time - start_time  # Cálculo del tiempo total

    # Guardar información en un archivo de texto
    with open("duracion_tiempo_ cant movi.txt", "a") as file:
        file.write(f"Inicio: {start_time:.6f} s\n")
        file.write(f"Fin: {end_time:.6f} s\n")
        file.write(f"Duración total: {duration:.6f} s\n")
        file.write("=" * 30 + "\n")

    print(f"Tiempo total de ejecución: {duration:.6f} segundos")
