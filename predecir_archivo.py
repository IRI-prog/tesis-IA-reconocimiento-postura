import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import  confusion_matrix
from collections import deque
# Importaciones de Keras
from tensorflow.keras.models import load_model
#from tensorflow.keras.utils import to_categorical

# Configuración inicial
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
COLORES = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def guardar_matriz_confusion_final(cm, folder_path, nombre_modelo, nombre_base):
    #Guarda la matriz de confusión con nombres de clases.
    # Definir nombres de las clases en el orden correcto (deben coincidir con label_encoder.classes_)
    nombres_clases = ["En Movimiento", "Parado", "Sentado", "Sin Personas"]
    
    plt.figure(figsize=(12, 8))  # Aumentar tamaño para mejor visualización
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap="Blues", annot_kws={"size": 14},
        cbar=False,
        xticklabels=nombres_clases,  # Etiquetas eje X
        yticklabels=nombres_clases    # Etiquetas eje Y
    )    
    # Ajustar etiquetas y títulos
    plt.title(f"Matriz de Confusión - {nombre_modelo}", fontsize=16, pad=20)
    plt.xlabel("Predicciones", fontsize=14, labelpad=15)
    plt.ylabel("Valores Reales", fontsize=14, labelpad=15)    
    # Rotar etiquetas para mejor legibilidad
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)    
    # Ajustar márgenes
    plt.tight_layout()    
    # Guardar
    plt.savefig(
        os.path.join(folder_path, f'Matriz_Confusion_{nombre_modelo}_{nombre_base}.png'),
        dpi=300, bbox_inches='tight')
    plt.close()
        
def extract_joints_from_txt(archivo_txt_path):
    try:
        with open(archivo_txt_path, "r") as archivo:
            lineas = archivo.readlines()
        coordenadas = []
        for linea in lineas:
            if "Person" not in linea:
                valores = [float(s) for s in linea.replace(",", " ").split()]
                if len(valores) >= 2:
                    coordenadas.append(valores)
        return np.array(coordenadas).flatten()
    except Exception as e:
        print(f"Error al extraer joints del archivo {archivo_txt_path}: {e}")
        return np.array([])

def buscar_archivos_csv(carpeta):
    print('buscar archivos csv')
    #Busca y retorna todos los archivos CSV en la carpeta dada.
    archivos_csv = []
    for root, dirs, files in os.walk(carpeta):
        for file in files:
            if file.endswith('.csv'):
                archivos_csv.append(os.path.join(root, file))
    return archivos_csv

def cargar_umbrales_json(ruta_archivo):
    #Carga los umbrales desde un archivo JSON.
    try:
        with open(ruta_archivo, 'r') as archivo:
            umbrales_str = json.load(archivo)
            # Convertir claves de string a enteros
            umbrales = {int(k): v for k, v in umbrales_str.items()}
        return umbrales
    except Exception as e:
        print(f"Error al cargar umbrales: {e}")
        return {}

def cargar_modelos(lstm_model_path, backpropagation_model_path):
    print('cargar los modelos')
    #Carga dos modelos desde las rutas especificadas.
    try:
        lstm_model = load_model(lstm_model_path)
        backpropagation_model = load_model(backpropagation_model_path)
        print("Modelos cargados exitosamente.")
        return lstm_model, backpropagation_model
    except Exception as e:
        print(f"Error al cargar los modelos: {e}")
        return None, None

def evaluar_modelos_completos(lstm_model, bp_model, label_encoder, archivo_prueba, umbrales_backpropagation, umbrales_lstm, carpeta_salida, nombre_base):
    try:
        print("🔮 Evaluando ambos modelos...")        
        # Cargar datos
        datos_prueba_df = pd.read_csv(archivo_prueba)        
       
        # Hacer copia para preservar datos originales
        df_final = datos_prueba_df.copy()        
        # Evaluar LSTM
        resultados_lstm, _ = predecir_y_evaluar(
            lstm_model, 
            label_encoder, 
            datos_prueba_df, 
            'LSTM', 
            carpeta_salida, 
            umbrales_lstm, nombre_base )
        # Añadir predicciones LSTM
        df_final['Prediccion LSTM'] = resultados_lstm['Prediccion Etiqueta']
        
        # Evaluar Backpropagation
        resultados_bp, _ = predecir_y_evaluar(
            bp_model, 
            label_encoder, 
            datos_prueba_df, 
            'Backpropagation', 
            carpeta_salida, 
            umbrales_backpropagation, nombre_base  )
        # Añadir predicciones Backpropagation
        df_final['Prediccion Backpropagation'] = resultados_bp['Prediccion Etiqueta']        
        # Guardar resultados combinados
        ruta_final = os.path.join(carpeta_salida, "resultados_combinados.csv")
        df_final.to_csv(ruta_final, index=False)
        print(f"\n✅ Archivo final guardado en: {ruta_final}")
        print("Columnas finales:", df_final.columns.tolist())        
        return df_final
            
    except Exception as e:
        print(f"🚨 Error en evaluación combinada: {str(e)}")
        return None

def generar_graficas_completas(resultados_df, label_encoder, carpeta_salida):
    #Genera gráficas comparativas y métricas detalladas para ambos modelos.
    try:
        print("\n📊 Generando gráficas comparativas...")
        
        # ===== 1. Distribución de predicciones =====
        plt.figure(figsize=(10, 6))
        sns.countplot(
            x='Prediccion_LSTM', 
            data=resultados_df, 
            order=label_encoder.classes_,
            color='blue',
            alpha=0.5,
            label='LSTM'
        )
        sns.countplot(
            x='Prediccion_Backpropagation', 
            data=resultados_df, 
            order=label_encoder.classes_,
            color='red',
            alpha=0.5,
            label='Backpropagation'
        )
        plt.title("Distribución de Predicciones por Modelo")
        plt.legend()
        plt.savefig(f"{carpeta_salida}/distribucion_predicciones.png", dpi=300)
        plt.close()
        
        # ===== 2. Comparación de confianzas =====
        plt.figure(figsize=(10, 6))
        sns.kdeplot(
            resultados_df['Confianza_LSTM'], 
            color='blue', 
            label='LSTM',
            fill=True
        )
        sns.kdeplot(
            resultados_df['Confianza_Backpropagation'], 
            color='red', 
            label='Backpropagation',
            fill=True
        )
        plt.title("Distribución de Confianzas por Modelo")
        plt.xlabel('Confianza')
        plt.legend()
        plt.savefig(f"{carpeta_salida}/distribucion_confianzas.png", dpi=300)
        plt.close()
        
        print("✅ Gráficas generadas exitosamente")
    
    except Exception as e:
        print(f"🚨 Error al generar gráficas: {str(e)}")

def preprocesar_datos(datos, columna_joints, longitud_secuencia, target_dim):
    """Procesa joints para crear secuencias temporales"""
    joints = []
    for path in datos[columna_joints]:
        try:
            j = extract_joints_from_txt(path)
            j = np.pad(j, (0, target_dim - len(j)), mode='constant')
        except:
            j = np.zeros(target_dim)
        joints.append(j)
    
    # Crear secuencias temporales
    secuencias = []
    for i in range(len(joints) - longitud_secuencia + 1):
        secuencias.append(joints[i:i+longitud_secuencia])
    
    return np.array(secuencias).reshape(-1, longitud_secuencia, 1)

def predecir_y_evaluar(modelo, label_encoder, datos_prueba, nombre_modelo, carpeta_salida, umbrales, nombre_base):
    print(f'🔍 Evaluando modelo {nombre_modelo}...')
    try:
        # 1. Validar datos
        if datos_prueba.empty:
            raise ValueError("❌ Datos de prueba vacíos")
        
        # 2. Procesar datos tabulares
        columnas_requeridas = [
            'Cantidad Movimiento', 'Promedio Actual', 'Promedio Siguiente',
            'Desviacion Actual', 'Desviacion Siguiente', 'Homogeneidad Actual',
            'Homogeneidad Siguiente', 'Velocidad Promedio', 'Simetria S_J']
        
        datos_principales = datos_prueba[columnas_requeridas]\
            .replace({'True': 1, 'False': 0})\
            .values.astype('float32')
        
        # 3. Procesar joints (función auxiliar)
        def procesar_joints_seguro(columna, target_dim):
            joints = []
            for path in datos_prueba[columna]:
                try:
                    # Asegurar que target_dim sea entero
                    target_dim_int = int(target_dim)
                    j = extract_joints_from_txt(path)[:target_dim_int]
                    j_padded = np.pad(j, (0, max(0, target_dim_int - len(j))), 'constant')
                    joints.append(j_padded)
                except Exception as e:
                    print(f"⚠️ Error procesando {path}: {str(e)}")
                    joints.append(np.zeros(target_dim_int))
            return np.array(joints)
        
        # 4. Obtener dimensiones requeridas
        # Obtener la dimensión directamente de la forma del input
        try:
            dim_actual = int(modelo.inputs[1].shape[1])  # Dimensión para joints actuales
            dim_siguiente = int(modelo.inputs[2].shape[1])  # Dimensión para joints siguientes
        except AttributeError:
            # Fallback si la forma no está disponible
            dim_actual = 34
            dim_siguiente = 34

        # 5. Procesar joints
        joints_actual = procesar_joints_seguro('Archivo Actual', dim_actual)
        joints_siguiente = procesar_joints_seguro('Archivo Siguiente', dim_siguiente)

        # 6. Generar predicciones
        print("\n🎯 Generando predicciones...")
        probabilidades = modelo.predict(
            [datos_principales, joints_actual, joints_siguiente],
            verbose=1, batch_size=32)
        
        # 7. Aplicar umbrales para determinar clases
        y_pred = []
        confianzas = []
        
        for prob in probabilidades:
            clase_seleccionada = None
            max_confianza = -1
            
            # Buscar clase que cumpla con los umbrales
            for clase_id, (umbral_min, umbral_max) in umbrales.items():
                confianza = prob[clase_id]
                if (umbral_min <= confianza <= umbral_max) and (confianza > max_confianza):
                    clase_seleccionada = clase_id
                    max_confianza = confianza
            
            # Si no cumple con ningún umbral, usar la de mayor probabilidad
            if clase_seleccionada is None:
                clase_seleccionada = np.argmax(prob)
                max_confianza = prob[clase_seleccionada]
            
            y_pred.append(clase_seleccionada)
            confianzas.append(max_confianza)
        
        # 8. Convertir códigos numéricos a etiquetas
        etiquetas_predichas = label_encoder.inverse_transform(y_pred)
        
        # 9. Construir resultados
        resultados_df = datos_prueba.copy()
        resultados_df[f'Prediccion_{nombre_modelo}'] = etiquetas_predichas
        resultados_df[f'Confianza_{nombre_modelo}'] = confianzas

        # 10. Guardar resultados
        os.makedirs(carpeta_salida, exist_ok=True)
        ruta_resultados = os.path.join(carpeta_salida, f'resultados_{nombre_base}_{nombre_modelo}.csv')
        resultados_df.to_csv(ruta_resultados, index=False)
        print(f"\n✅ Predicciones {nombre_modelo} guardadas en: {ruta_resultados}")
        
        return resultados_df
        
    except Exception as e:
        print(f"🚨 Error en {nombre_modelo}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

        
def predecir_y_evaluar_lstm(modelo, label_encoder, datos_prueba, nombre_modelo, carpeta_salida, umbrales, nombre_base):
    print('🔍 Evaluando modelo LSTM con manejo temporal...')
    longitud_secuencia= 9 
    try:
        # ===== 1. Validación inicial =====
        if datos_prueba.empty:
            raise ValueError("❌ Datos de prueba vacíos")
        # Procesar datos tabulares
    
        columnas_requeridas = [
            'Cantidad Movimiento', 'Promedio Actual', 'Promedio Siguiente',
            'Desviacion Actual', 'Desviacion Siguiente', 'Homogeneidad Actual',
            'Homogeneidad Siguiente', 'Velocidad Promedio', 'Simetria S_J']
        datos_procesados = datos_prueba.copy()
        
        # Convertir columnas que puedan contener booleanos
        for col in columnas_requeridas:
            if datos_procesados[col].dtype == bool:
                datos_procesados[col] = datos_procesados[col].astype(int)
            elif datos_procesados[col].dtype == object:
                # Reemplazar strings de booleanos
                datos_procesados[col] = datos_procesados[col].replace({
                    'True': 1, 'False': 0,
                    'true': 1, 'false': 0
                })
                # Convertir a numérico (maneja otros strings como NaN)
                datos_procesados[col] = pd.to_numeric(datos_procesados[col], errors='coerce')
        
        # Rellenar NaN resultantes de la conversión
        datos_procesados[columnas_requeridas] = datos_procesados[columnas_requeridas].fillna(0)        
        X_principal = datos_procesados[columnas_requeridas].values.astype('float32')
        
        # Procesar joints con verificación
        print("\n🔍 Ejemplo de joints actuales (primer sample):")
        X_joints_actual = preprocesar_datos(datos_prueba, 'Archivo Actual', 34, 34)
        X_joints_siguiente = preprocesar_datos(datos_prueba, 'Archivo Siguiente', 34, 34)
        if len(X_joints_actual) > 0:
            print(X_joints_actual[0][:5])  # Primeros 5 pasos temporales

        # Forzar dimensión correcta para datos cero
        if X_principal.shape[0] == 0:
            X_principal = np.zeros((1, longitud_secuencia, 9))
        if X_joints_actual.shape[0] == 0:
            X_joints_actual = np.zeros((1, 34, 1))
        if X_joints_siguiente.shape[0] == 0:
            X_joints_siguiente = np.zeros((1, 34, 1))

        # Ajustar cardinalidad
        min_muestras = min(len(X_principal), len(X_joints_actual), len(X_joints_siguiente))
        X_principal = X_principal[:min_muestras]
        X_joints_actual = X_joints_actual[:min_muestras]
        X_joints_siguiente = X_joints_siguiente[:min_muestras]
        
        # ===== 2. Predicción =====
        print("\n🎯 Generando predicciones temporales...")
        probabilidades = modelo.predict(
            [X_principal, X_joints_actual, X_joints_siguiente],
            verbose=1, batch_size=32)
        
        # ===== 3. Post-procesamiento con umbrales mejorados =====
        y_pred = []
        confianzas = []
        buffer_secuencia = []  # Almacena las últimas N probabilidades para contexto temporal
        
        for i, prob in enumerate(probabilidades):
            # 1. Mantener un buffer de la secuencia actual
            buffer_secuencia.append(prob)
            if len(buffer_secuencia) > longitud_secuencia:
                buffer_secuencia.pop(0)
            
            # 2. Calcular probabilidad ponderada temporalmente
            pesos = np.linspace(0.3, 1.0, len(buffer_secuencia))  # Mayor peso a muestras recientes
            prob_ponderada = np.average(buffer_secuencia, axis=0, weights=pesos)            
            # 3. Determinar umbrales dinámicos basados en la secuencia ACTUAL
            umbral_min = np.percentile(prob_ponderada, 25)
            umbral_max = np.percentile(prob_ponderada, 75)            
            # 4. Selección de clase con lógica secuencial
            clase_seleccionada = None
            max_confianza = -1
            
            # Primero: Buscar clases que cumplan umbrales dinámicos
            for clase_id in umbrales.keys():
                confianza = prob_ponderada[clase_id]
                if confianza >= umbral_min and confianza >= umbrales[clase_id][0]:
                    if confianza > max_confianza:
                        max_confianza = confianza
                        clase_seleccionada = clase_id            
            # Fallback: Usar máximo histórico con memoria temporal
            if clase_seleccionada is None:
                avg_historico = np.mean([p for p in buffer_secuencia], axis=0)
                clase_seleccionada = np.argmax(avg_historico)
                max_confianza = avg_historico[clase_seleccionada]            
            # 5. Suavizado temporal de predicciones
            if len(y_pred) > 0:
                # Considerar la predicción anterior para transiciones suaves
                if max_confianza < 0.7 and y_pred[-1] == clase_seleccionada:
                    clase_seleccionada = y_pred[-1]  # Mantener clase anterior si confianza baja                    
            y_pred.append(clase_seleccionada)
            confianzas.append(max_confianza)
                
        # ===== 4. Construcción de resultados =====
        etiquetas = label_encoder.inverse_transform(y_pred)
                
        # ===== 9. Construcción del DataFrame =====
        start_idx = max(0, len(datos_prueba) - len(y_pred))
        resultados_df = datos_prueba.iloc[start_idx:].copy()
        resultados_df[f'Prediccion_{nombre_modelo}'] = etiquetas
        resultados_df[f'Confianza_{nombre_modelo}'] = confianzas

        # ===== 10. Guardado seguro =====
        os.makedirs(carpeta_salida, exist_ok=True)
        ruta_resultados = os.path.join(carpeta_salida, f'resultados_{nombre_modelo}.csv')
        resultados_df.to_csv(ruta_resultados, index=False)
        
        print(f"\n✅ Predicciones LSTM guardadas en: {ruta_resultados}")
        return resultados_df

    except Exception as e:
        print(f"🚨 Error crítico en evaluación LSTM: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Retornar DataFrame vacío para evitar None

        
def evaluar_modelos_completos(lstm_model, bp_model, label_encoder, archivo_prueba, umbrales_backpropagation, umbrales_lstm, carpeta_salida):
    try:
        print("🔮 Evaluando ambos modelos...")
        # Cargar datos
        datos_prueba_df = pd.read_csv(archivo_prueba)
        nombre_base = os.path.splitext(os.path.basename(archivo_prueba))[0]
        
        # Hacer copia para preservar datos originales
        df_final = datos_prueba_df.copy()
        
        # Evaluar modelos
        resultados_lstm = predecir_y_evaluar_lstm(
            lstm_model, label_encoder, datos_prueba_df,
            'LSTM', carpeta_salida, umbrales_lstm, nombre_base)
        
        resultados_bp = predecir_y_evaluar(
            bp_model, label_encoder, datos_prueba_df,
            'Backpropagation', carpeta_salida, umbrales_backpropagation, nombre_base)
        
        # Combinar resultados
        if resultados_lstm is not None:
            df_final = df_final.merge(
                resultados_lstm[[f'Prediccion_LSTM', f'Confianza_LSTM']],
                left_index=True,
                right_index=True
            )
        
        if resultados_bp is not None:
            df_final = df_final.merge(
                resultados_bp[[f'Prediccion_Backpropagation', f'Confianza_Backpropagation']],
                left_index=True,
                right_index=True
            )
        
        # Guardar resultados combinados
        ruta_final = os.path.join(carpeta_salida, f"{nombre_base}_resultados_combinados.csv")
        df_final.to_csv(ruta_final, index=False)
        print(f"\n✅ Archivo final guardado en: {ruta_final}")
        return df_final
            
    except Exception as e:
        print(f"🚨 Error en evaluación combinada: {str(e)}")
        return None

def funcionPrincipal(archivo_nuevo, base_directory):#archivo_salida, base_directory
    print("Empezando en la clase de predecir")
    print("archivo_nuevo", archivo_nuevo)
    print("base_directory", base_directory)
    label_encoder = LabelEncoder()
    label_encoder.fit(["En Movimiento", "Parado", "Sentado", "Sin Personas"])    
    # Ejemplo de uso 
    #umbrales_lstm = cargar_umbrales_json('D:/programs/redNeuronal/modificaciones/datos agrupar/modelos_entrenados/modelo_LSTM/umbrales 01_umbrales.json')
    umbrales_lstm = cargar_umbrales_json('D:/programs/redNeuronal/modificaciones/prueba/modelo 19-04/umbrales 12_umbrales.json')
    print("umbrales_lstm", umbrales_lstm)
    umbrales_backpropagation =cargar_umbrales_json('D:/programs/redNeuronal/modificaciones/prueba/modelo 19-04/back/modelo_backpropagation/umbrales 02_umbrales.json')
    print("umbrales_backpropagation", umbrales_backpropagation)
    # Llamada a la función para hacer predicciones
    lstm_model_path="D:/programs/redNeuronal/modificaciones/prueba/modelo 19-04/modelo_LSTM_version 12_modelo_LSTM.keras"
    #lstm_model_path = 'D:/programs/redNeuronal/modificaciones/datos agrupar/modelos_entrenados/modelo_LSTM/modelo_LSTM 03_modelo_LSTM.keras'
    backpropagation_model_path = 'D:/programs/redNeuronal/modificaciones/prueba/modelo 19-04/back/modelo_backpropagation 02_modelo_Back.keras'

    modelo_lstm, modelo_backprop = cargar_modelos(lstm_model_path, backpropagation_model_path)
    #archivo_nuevo = 'D:/programs/redNeuronal/modificaciones/datos_agrupar/seguimiento/video_20250311_125501_recorte_93-160.csv'
    # Evaluación completa (pasar la ruta del CSV)
    #carpeta_resultados = "seguimiento/entrenamiento y prueba/seguimiento"
    
    # 2. Ejecutar evaluación completa
    evaluar_modelos_completos(
        modelo_lstm, 
        modelo_backprop,
        label_encoder,
        archivo_nuevo,  
        umbrales_backpropagation,
        umbrales_lstm, 
        base_directory )
    
    
    
if __name__ == "__main__":
    # Crear y ajustar el LabelEncoder
    import time
    start_time = time.time()  # Iniciar temporizador

    #cominacion de todos los datos
    #archivo_nuevo='D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet/evaluar_prediccion/resultados_consolidados/datos_consolidados.csv'
    #archivo_nuevo="D:/programs/redNeuronal/modificaciones/programa_final/datos/cantidadMovimiento/segmentos Persona 2 video_20240214_123653_2.csv"
    #base_directory="D:/programs/redNeuronal/modificaciones/programa_final/datos"
    
    #funcionPrincipal(archivo_nuevo, base_directory)
    funcionPrincipal()    
    end_time = time.time()  # Fin del temporizador
    duration = end_time - start_time  # Cálculo del tiempo total

    # Guardar información en un archivo de texto
    with open("duracion_tiempo_ predecirEvaluar.txt", "a") as file:
        file.write(f"Inicio: {start_time:.6f} s\n")
        file.write(f"Fin: {end_time:.6f} s\n")
        file.write(f"Duración total: {duration:.6f} s\n")
        file.write("=" * 30 + "\n")

    print(f"Tiempo total de ejecución: {duration:.6f} segundos")
    
