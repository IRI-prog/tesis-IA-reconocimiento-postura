"""
import os
import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Importaciones de Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape, Dropout
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Accuracy, Precision, Recall, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import SparseCategoricalAccuracy 
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Flatten, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.callbacks import Callback

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import label_binarize


class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, joints_1_test, joints_2_test, y_test, folder_path, num_classes=4):
        super().__init__()
        self.x_test = x_test
        self.joints_1_test = joints_1_test
        self.joints_2_test = joints_2_test
        self.y_test = y_test
        self.folder_path = folder_path
        self.num_classes = num_classes
        self.epoch_counter = 0
        self.confusion_matrices = []
        
        # Inicializar estructura para almacenar métricas
        self.metrics_history = {
            "epoch": [],
            **{f"precision_{i}": [] for i in range(num_classes)},
            **{f"recall_{i}": [] for i in range(num_classes)},
            **{f"f1_{i}": [] for i in range(num_classes)},
            "accuracy": [],
            "f1_score": [],
            "recall": [],
            "mse": [],
            "mae": [],
            "r2": [],
            "correlation": [],
            "confusion_matrix": []
        }
        
        # Configurar CSV
        self.csv_path = os.path.join(folder_path, "metricas_por_epoca_LSTM.csv")
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.metrics_history.keys()).to_csv(self.csv_path, index=False)        
        # Preparar carpetas para gráficas
        self.preparar_directorios()
        

    def preparar_directorios(self):
        #Crea carpetas para guardar gráficas y matrices de confusión.
        self.reg_plots_dir = os.path.join(self.folder_path, "regression_metrics_plots")
        self.cm_plots_dir = os.path.join(self.folder_path, "confusion_matrices")
        os.makedirs(self.reg_plots_dir, exist_ok=True)
        os.makedirs(self.cm_plots_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_counter += 1
        
        # Paso 1: Predecir con el modelo
        y_pred = self.model.predict(
            [self.x_test, self.joints_1_test, self.joints_2_test], verbose=0 )
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = self.y_test
        
        # Paso 2: Convertir etiquetas a one-hot para métricas de regresión
        y_true_onehot = label_binarize(y_true_classes, classes=np.arange(self.num_classes))
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        # Paso 3: Calcular métricas globales
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        mse = mean_squared_error(y_true_onehot, y_pred)
        mae = mean_absolute_error(y_true_onehot, y_pred)
        r2 = r2_score(y_true_onehot.flatten(), y_pred.flatten())
        correlation = np.corrcoef(y_true_onehot.flatten(), y_pred.flatten())[0, 1]
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        # Calcular matriz de confusión
        
         # Paso 4: Calcular métricas por clase - FORZAR TODAS LAS CLASES
        report = classification_report(
            y_true_classes,
            y_pred_classes,
            labels=np.arange(self.num_classes),  # <--- ¡Añadir esto!
            target_names=[f"Class_{i}" for i in range(self.num_classes)],
            output_dict=True,
            zero_division=0 )        
    
        # Paso 5: Extraer métricas asegurando todas las clases
        class_metrics = {}
        for i in range(self.num_classes):
            class_name = f"Class_{i}"
            class_report = report.get(class_name, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})            
            class_metrics[f"precision_{i}"] = class_report["precision"]
            class_metrics[f"recall_{i}"] = class_report["recall"]
            class_metrics[f"f1_{i}"] = class_report["f1-score"]
    
        # Paso 6: Crear DataFrame con orden de columnas consistente
        df_data = {"epoch": epoch + 1}        
        # Añadir métricas en orden numérico
        for i in range(self.num_classes):
            df_data[f"precision_{i}"] = class_metrics[f"precision_{i}"]
            df_data[f"recall_{i}"] = class_metrics[f"recall_{i}"]
            df_data[f"f1_{i}"] = class_metrics[f"f1_{i}"]
        
        # Añadir métricas globales
        df_data.update({
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "correlation": correlation,
            "confusion_matrix": np.array2string(cm, separator="|")
            })
    
        # Crear DataFrame con las columnas en orden correcto
        column_order = ["epoch"] + \
                       [f"precision_{i}" for i in range(self.num_classes)] + \
                       [f"recall_{i}" for i in range(self.num_classes)] + \
                       [f"f1_{i}" for i in range(self.num_classes)] + \
                       ["accuracy", "mse", "mae", "r2", "correlation", "confusion_matrix"]
        
        # Crear DataFrame con las columnas en orden correcto
        column_order = ["epoch"] + \
                    [f"precision_{i}" for i in range(self.num_classes)] + \
                    [f"recall_{i}" for i in range(self.num_classes)] + \
                    [f"f1_{i}" for i in range(self.num_classes)] + \
                    ["macro_f1", "macro_recall", "accuracy", "mse", "mae", "r2", "correlation", "confusion_matrix"]
        df = pd.DataFrame([df_data], columns=column_order)
    
        # Guardar en CSV
        header = not os.path.exists(self.csv_path)  # Escribir header solo
        
        # Paso 6: Actualizar el historial de métricas
        self.metrics_history["epoch"].append(epoch + 1)
        for metric in class_metrics:
            self.metrics_history[metric].append(class_metrics[metric])
        self.metrics_history["accuracy"].append(accuracy)
        self.metrics_history["mse"].append(mse)
        self.metrics_history["mae"].append(mae)
        # Actualizar metrics_history:
        self.metrics_history["f1_score"].append(f1)
        self.metrics_history["recall"].append(recall)

        self.metrics_history["r2"].append(r2)
        self.metrics_history["correlation"].append(correlation)
        self.metrics_history["confusion_matrix"].append(np.array2string(cm, separator="|"))
        self.confusion_matrices.append(cm)  
        
        # Paso 7: Guardar en CSV
        df_data = {
            "epoch": epoch + 1,
            **class_metrics,
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "correlation": correlation,
            "confusion_matrix": np.array2string(cm, separator="|") }
        df = pd.DataFrame([df_data])
        df.to_csv(self.csv_path, mode="a", header=False, index=False)
        
        # Paso 8: Generar gráficas y matrices
        self.generar_graficas_regresion()
        self.guardar_matriz_confusion(cm, epoch + 1)      
        # Guardar matriz final
        self.guardar_matriz_confusion_final(cm)        
        # Opcional: Guardar también en formato CSV
        np.savetxt(os.path.join(self.folder_path, "confusion_matrix_final.csv"), 
                cm, fmt="%d", delimiter=",")
        
        # Paso 9: Log en consola
        print(f"\nÉpoca {epoch + 1}:")
        print(f"  - Precisión Global: {accuracy:.4f}")
        print(f"  - F1-Score (Clase 0): {class_metrics['f1_0']:.4f}")

    def generar_graficas_regresion(self):
        #Genera gráficas de evolución de MSE, MAE, R² y Correlación.
        epochs = self.metrics_history["epoch"]
        for metric in ["mse", "mae", "r2", "correlation"]:
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.metrics_history[metric], "b-", label=metric)
            plt.title(f"Evolución de {metric.upper()} por Época")
            plt.xlabel("Época")
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.reg_plots_dir, f"{metric}.png"))
            plt.close()
    
    def guardar_matriz_confusion(self, cm, epoch):
        #Guarda la matriz de confusión como imagen PNG.
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión - Época {epoch}")
        plt.xlabel("Predicciones")
        plt.ylabel("Reales")
        plt.savefig(os.path.join(self.cm_plots_dir, f"cm_epoch_{epoch:03d}.png"))
        plt.close()
    
    def guardar_matriz_confusion_final(self, cm):
        #Guarda la matriz de confusión final del conjunto de prueba.
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    annot_kws={"size": 12}, cbar=False)
        plt.title("Matriz de Confusión - Conjunto de Prueba Final", fontsize=14)
        plt.xlabel("Predicciones", fontsize=12)
        plt.ylabel("Valores Reales", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Guardar en alta resolución
        plt.savefig(os.path.join(self.folder_path, "confusion_matrix_final.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
     
# Función para buscar archivos CSV
def buscar_archivos_csv(carpeta):
    archivos_csv = []
    for root, dirs, files in os.walk(carpeta):
        for file in files:
            if file.endswith('.csv'):
                ruta_completa = os.path.join(root, file)
                # Obtener la etiqueta desde la carpeta padre directa
                etiqueta = os.path.basename(os.path.dirname(ruta_completa))
                archivos_csv.append((ruta_completa, etiqueta))
    return archivos_csv

 #Función para extraer coordenadas de joints desde archivos TXT

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

def normalizar_joints(joints):
    if joints.shape[0] > 0:
        scaler = MinMaxScaler()
        original_shape = joints.shape
        joints_flat = joints.reshape(-1, joints.shape[-1])
        joints_normalized = scaler.fit_transform(joints_flat)
        return joints_normalized.reshape(original_shape)
    return joints

# Cargar datos y etiquetas
def cargar_datos_por_categoria(ruta_archivos):
    try:
        print('Empieza a cargar datos')
        datos_completos = []
        etiquetas_str = []
        joint_coords_1 = []
        joint_coords_2 = []
        all_coords = []
        
        # 1. Configurar diccionario de etiquetas
        unique_labels = set(ruta_archivos.values())
        labels_dict = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print("Diccionario de etiquetas:", labels_dict)

        # 2. Procesar archivos
        for archivo, etiqueta in ruta_archivos.items():
            if not os.path.exists(archivo):
                continue
                
            datos = pd.read_csv(archivo, encoding='ISO-8859-1')            
            # Validar columnas
            columnas_requeridas = [
                "Cantidad Movimiento", "Promedio Actual", "Promedio Siguiente",
                "Desviacion Actual", "Desviacion Siguiente", 
                "Homogeneidad Actual", "Homogeneidad Siguiente", 
                "Simetria S_J", "Velocidad Promedio"]
            if not all(col in datos.columns for col in columnas_requeridas):
                print(f"¡Archivo {archivo} ignorado: columnas faltantes!")
                continue
            # Añade conversión numérica:
            datos[columnas_requeridas] = datos[columnas_requeridas].apply(pd.to_numeric, errors='coerce')            
            # Luego elimina filas con NaN (si es apropiado):
            datos = datos.dropna(subset=columnas_requeridas)
            
            # Extraer datos principales
            datos_seleccionados = datos[columnas_requeridas].values
            if len(datos_seleccionados) == 0:
                print(f"¡Archivo {archivo} ignorado: sin datos válidos!")
                continue

            # Procesar coordenadas
            joint_paths_1 = datos['Archivo Actual'].values
            joint_paths_2 = datos['Archivo Siguiente'].values
            
            # Asegurar arrays 2D y almacenar
            coords_1 = [_ensure_2d(extract_joints_from_txt(path)) for path in joint_paths_1]
            coords_2 = [_ensure_2d(extract_joints_from_txt(path)) for path in joint_paths_2]
            
            joint_coords_1.extend(coords_1)
            joint_coords_2.extend(coords_2)
            all_coords.extend(joint_coords_1 + joint_coords_2)
            
            datos_completos.append(datos_seleccionados)
            etiquetas_str.extend([etiqueta] * len(datos_seleccionados))

        # 3. Calcular dimensiones máximas
        max_length = max(len(seq) for seq in all_coords)
        max_features = max(np.array(seq).shape[1] for seq in joint_coords_1 + joint_coords_2)

        #max_features = max(seq.shape[1] for seq in joint_coords_1 + joint_coords_2)  # Ahora seguro que son 2D
        print(f"Longitud máxima temporal: {max_length}")
        print(f"Características máximas: {max_features}")

        # 4. Aplicar padding
        joint_coords_1 =[np.pad(seq, ((0, max_length - len(seq)), (0, max_features - seq.shape[1])), mode="constant") for seq in joint_coords_1]      
        joint_coords_2 = [np.pad(seq, ((0, max_length - len(seq)), (0, max_features - seq.shape[1])), mode="constant") for seq in joint_coords_2]
        
        # 5. Convertir a arrays numpy
        joint_coords_1 = np.array(joint_coords_1)
        joint_coords_2 = np.array(joint_coords_2)

        # 6. Normalizar joints (ahora son arrays numpy)
        #joint_coords_1 = normalizar_joints(joint_coords_1)
        #joint_coords_2 = normalizar_joints(joint_coords_2)
        
        # 5. Convertir a arrays numpy
        datos_completos = np.vstack(datos_completos)
        joint_coords_1 = np.array(joint_coords_1)
        joint_coords_2 = np.array(joint_coords_2)
        etiquetas = np.array([labels_dict[label] for label in etiquetas_str], dtype=np.int32)
        
        # Verificación final
        print("\nFormas finales:")
        print(f"- Datos: {datos_completos.shape}")
        print(f"- Joints 1: {joint_coords_1.shape}")
        print(f"- Joints 2: {joint_coords_2.shape}")
        print(f"- Etiquetas: {etiquetas.shape}")        
        return datos_completos, etiquetas, joint_coords_1, joint_coords_2
        
    except Exception as e:
        print(f"¡Error crítico!: {e}")
        return None, None, None, None

# Función auxiliar para asegurar arrays 2D
def _ensure_2d(array):
    if array.ndim == 1:
        return array.reshape(-1, 1)  # Convertir (n,) -> (n, 1)
    return array
       
def get_next_filename(folder_path, base_name):
    output_base_folder = os.path.join(folder_path, 'modelo_LSTM_version')
    os.makedirs(output_base_folder, exist_ok=True)
    
    base_name = base_name.strip() + " "  # Asegurar espacio al final
    pattern = re.compile(rf"{re.escape(base_name)}(\d{{2}})_modelo_LSTM.keras")  # Escapar puntos
    
    max_num = 0
    for file in os.listdir(output_base_folder):
        match = pattern.match(file)
        if match:
            max_num = max(max_num, int(match.group(1)))
    
    next_num = max_num + 1
    new_path = os.path.join(output_base_folder, f"{base_name}{next_num:02d}_modelo_LSTM.keras")
    
    # Prevenir colisiones (por si acaso)
    while os.path.exists(new_path):
        next_num += 1
        new_path = os.path.join(output_base_folder, f"{base_name}{next_num:02d}_modelo_LSTM.keras")
    
    return new_path, next_num

#Guardar las metricas en una imagen
def save_metrics_image(folder_path, base_name, figure):
    print('empieza a guardar metricas')
    output_base_folder = os.path.join(folder_path, 'Metricas_imagenes_version') 
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
        print(f"Se ha creado la carpeta: {output_base_folder}")
    # Filtrar archivos que empiezan con base_name y extraer números correctamente
    existing_files = [f for f in os.listdir(output_base_folder) if f.startswith(base_name)]
    max_num = 0
    for file in existing_files:
        try:
            num_part = file[len(base_name):].strip().split('_')[0]  # Extraer número correctamente
            num = int(num_part)
            max_num = max(max_num, num)
        except (ValueError, IndexError):
            continue  # Si hay un error en la conversión, se ignora el archivo
    next_num = max_num + 1
    image_name = os.path.join(output_base_folder, f"{base_name} {next_num:02d}_modelo_LSTM.png")    
   # Verificar si la figura es válida antes de guardarla
    if isinstance(figure, plt.Figure):
        figure.savefig(image_name)
        plt.close(figure)
        return image_name
    else:
        raise ValueError("El objeto figure no es una instancia válida de matplotlib.figure.Figure")

def graficar_evolucion_matrices(metrics_callback, folder_path):
    plt.figure(figsize=(12, 8))    
    output_base_folder = os.path.join(folder_path, 'Evolucion_cm_epoch_version')
    for i, cm in enumerate(metrics_callback.confusion_matrices):
        plt.clf()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión - Época {i+1}")
        plt.xlabel("Predicciones")
        plt.ylabel("Reales")
        plt.savefig(os.path.join(output_base_folder, f"evolucion_cm_epoch_{i+1:03d}.png"))
        plt.close()
    
def graficar_metricas(metrics_callback, history, image_path):
    print('Empieza a graficar métricas')    
    epochs = range(1, len(metrics_callback.metrics_history["epoch"]) + 1)
    
    # Configurar grid de 6x2
    fig = plt.figure(figsize=(20, 30))
    gs = fig.add_gridspec(6, 2, height_ratios=[1.5,1.5,1.5,1,1,1])
    
    # Subplots principales
    ax0 = fig.add_subplot(gs[0, 0])  # F1 por clase
    ax1 = fig.add_subplot(gs[0, 1])  # Accuracy
    ax2 = fig.add_subplot(gs[1, 0])  # Loss
    ax3 = fig.add_subplot(gs[1, 1])  # Precisión por clase
    ax4 = fig.add_subplot(gs[2, 0])  # Recall por clase
    ax5 = fig.add_subplot(gs[2, 1])  # MSE
    
    # Mini subplots para métricas macro
    ax6 = fig.add_subplot(gs[3, 0])  # Macro F1
    ax7 = fig.add_subplot(gs[3, 1])  # Macro Recall
    ax8 = fig.add_subplot(gs[4, 0])  # MAE
    ax9 = fig.add_subplot(gs[4, 1])  # R²
    ax10 = fig.add_subplot(gs[5, 0]) # Correlación
    
    # 1. F1 por clase (Original mejorado)
    colors = plt.cm.tab20(np.linspace(0, 1, metrics_callback.num_classes))
    for i in range(metrics_callback.num_classes):
        ax0.plot(epochs, metrics_callback.metrics_history[f"f1_{i}"], 
                label=f'Clase {i}', color=colors[i], linewidth=1.5)
    ax0.set_title('Evolución de F1-Score por Clase', fontsize=12)
    ax0.legend(ncol=2, fontsize=8)
    ax0.grid(alpha=0.2)

    # 2. Accuracy (Original)
    ax1.plot(history.epoch, history.history['accuracy'], 'navy', label='Entrenamiento')
    ax1.plot(history.epoch, history.history['val_accuracy'], 'skyblue', label='Validación')
    ax1.set_title('Precisión Global', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.2)

    # 3. Loss (Original)
    ax2.plot(history.epoch, history.history['loss'], 'darkred', label='Entrenamiento')
    ax2.plot(history.epoch, history.history['val_loss'], 'lightcoral', label='Validación')
    ax2.set_title('Pérdida', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.2)

    # 4. Precisión por clase (Original)
    for i in range(metrics_callback.num_classes):
        ax3.plot(epochs, metrics_callback.metrics_history[f"precision_{i}"], 
                color=colors[i], linestyle='--', alpha=0.7)
    ax3.set_title('Precisión por Clase', fontsize=12)
    ax3.grid(alpha=0.2)

    # 5. Recall por clase (Original)
    for i in range(metrics_callback.num_classes):
        ax4.plot(epochs, metrics_callback.metrics_history[f"recall_{i}"], 
                color=colors[i], linestyle='-.', alpha=0.7)
    ax4.set_title('Recall por Clase', fontsize=12)
    ax4.grid(alpha=0.2)

    # 6. Métricas de regresión (Original)
    ax5.plot(epochs, metrics_callback.metrics_history["mse"], 'purple')
    ax5.set_title('MSE', fontsize=12)
    ax5.grid(alpha=0.2)
    
    # 7. Nuevo: Macro F1
    ax6.plot(epochs, metrics_callback.metrics_history["f1_score"], 'green', linewidth=2)
    ax6.set_title('F1-Score Macro', fontsize=10)
    ax6.set_ylim(0, 1)
    ax6.grid(alpha=0.2)

    # 8. Nuevo: Macro Recall
    ax7.plot(epochs, metrics_callback.metrics_history["recall"], 'blue', linewidth=2)
    ax7.set_title('Recall Macro', fontsize=10)
    ax7.set_ylim(0, 1)
    ax7.grid(alpha=0.2)

    # Resto de métricas de regresión
    ax8.plot(epochs, metrics_callback.metrics_history["mae"], 'orange')
    ax8.set_title('MAE', fontsize=12)
    ax8.grid(alpha=0.2)
    
    ax9.plot(epochs, metrics_callback.metrics_history["r2"], 'red')
    ax9.set_title('R²', fontsize=12)
    ax9.grid(alpha=0.2)
    
    ax10.plot(epochs, metrics_callback.metrics_history["correlation"], 'magenta')
    ax10.set_title('Correlación', fontsize=12)
    ax10.grid(alpha=0.2)

    plt.tight_layout()
    #plt.savefig(image_path, dpi=300, bbox_inches='tight')
    #plt.close()
    
    # Guardar imagen
    image_path_specific = save_metrics_image(folder_path, "metricas_combinadas", fig)
    #image_path_specific = os.path.join(folder_path, 'metricas_combinadas.png')
    plt.savefig(image_path_specific, dpi=300, bbox_inches='tight')
    plt.close()
    return image_path_specific

# Métrica personalizada para Sensibilidad (Recall positivo)
def sensibilidad(y_true, y_pred):
    return tf.keras.metrics.Recall()(y_true, y_pred)

def ajustar_dimensiones(x, expected_timesteps, expected_features):
    x = np.asarray(x, dtype=np.float32)    
    # Eliminar dimensiones de tamaño 1 (excepto batch)
    while len(x.shape) > 3 and x.shape[-1] == 1:
        x = np.squeeze(x, axis=-1)    
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=1)  # Añadir dimensión temporal    
    # Ajustar timesteps y features
    current_timesteps = x.shape[1]
    current_features = x.shape[2]    
    # Padding/Truncado para timesteps
    if current_timesteps < expected_timesteps:
        pad = ((0, 0), (0, expected_timesteps - current_timesteps), (0, 0))
        x = np.pad(x, pad, mode='constant')
    elif current_timesteps > expected_timesteps:
        x = x[:, :expected_timesteps, :]    
    # Padding/Truncado para features
    if current_features < expected_features:
        pad = ((0, 0), (0, 0), (0, expected_features - current_features))
        x = np.pad(x, pad, mode='constant')
    elif current_features > expected_features:
        x = x[:, :, :expected_features]    
    return x

def construir_LSTM(timesteps, num_feature):
    try:
        # Rama para métricas (serie temporal)
        input_metrics = Input(shape=(timesteps, 1), name='metrics_input')
        lstm_metrics = LSTM(128, return_sequences=False)(input_metrics)
        dense_metrics = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(lstm_metrics)  
        
        # Rama para keypoints de la imagen actual
        input_joint1 = Input(shape=(num_feature, 1), name='input_joint1')
        lstm_joint1 = LSTM(128, return_sequences=False)(input_joint1)
        dense_joint1 = Dense(64, activation='relu')(lstm_joint1)
        
        # Rama para keypoints de la imagen siguiente
        input_joint2 = Input(shape=(num_feature, 1), name='input_joint2')
        lstm_joint2 = LSTM(128, return_sequences=False)(input_joint2)
        dense_joint2 = Dense(64, activation='relu')(lstm_joint2)
        
        # Concatenación y fusión
        combinado = Concatenate()([dense_metrics, dense_joint1, dense_joint2])
        combinado = Dense(64, activation='relu')(combinado)
        combinado = Dropout(0.3)(combinado)
        
        # Salida
        salida = Dense(4, activation='softmax')(combinado)        
        modelo = Model(inputs=[input_metrics, input_joint1, input_joint2], outputs=salida)
        modelo.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        modelo.summary()
        return modelo
    except Exception as e:
        print(f"Error al construir el modelo: {e}")
        return None

# Entrenar el modelo LSTM
def entrenar_LSTM(modelo, x_train, joints_1_train, joints_2_train, y_train, 
                  x_test, joints_1_test, joints_2_test, y_test, folder_path):
    try:
       # --- Ajustar dimensiones y eliminar 4D ---
        # Para datos de entrenamiento
        x_train = ajustar_dimensiones(x_train, 9, 1)
        joints_1_train = np.squeeze(joints_1_train, axis=-1)  # <--- Primero squeeze
        joints_1_train = ajustar_dimensiones(joints_1_train, 34, 1)
        joints_2_train = np.squeeze(joints_2_train, axis=-1)  # <--- Primero squeeze
        joints_2_train = ajustar_dimensiones(joints_2_train, 34, 1)
        
        # Para datos de prueba
        x_test = ajustar_dimensiones(x_test, 9, 1)
        joints_1_test = np.squeeze(joints_1_test, axis=-1)    # <--- Primero squeeze
        joints_1_test = ajustar_dimensiones(joints_1_test, 34, 1)
        joints_2_test = np.squeeze(joints_2_test, axis=-1)    # <--- Primero squeeze
        joints_2_test = ajustar_dimensiones(joints_2_test, 34, 1)
        
        print(f"Forma de y_train: {y_train.shape}")
        print(f"Forma de y_test: {y_test.shape}")
        print(f"x_test shape: {x_test.shape}")
        print(f"joints_1_test shape: {joints_1_test.shape}")
        print(f"joints_2_test shape: {joints_2_test.shape}")

        # Crear el callback de métricas
        metrics_callback = MetricsCallback(x_test, joints_1_test, joints_2_test, y_test, folder_path, num_classes=4)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)  
        
        # Entrenar el modelo
        history = modelo.fit(
            [x_train, joints_1_train, joints_2_train], y_train,
            validation_data=([x_test, joints_1_test, joints_2_test], y_test),
            epochs=100, batch_size=32, verbose=2, 
            callbacks=[metrics_callback, early_stop])        
        
        # Evaluar las métricas finales en el conjunto de prueba
        y_pred = modelo.predict([x_test, joints_1_test, joints_2_test])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = y_test  
        print("Etiquetas únicas en y_train:", np.unique(y_train))  

        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        accuracy = accuracy_score(y_true_classes, y_pred_classes) 

        cm = confusion_matrix(y_true_classes, y_pred_classes)   
        cm_final = confusion_matrix(y_test, y_pred_classes)  # ¡Sin np.argmax!  
        # Guardar todas las métricas en un CSV
        df_metrics = pd.DataFrame(metrics_callback.metrics_history)
        csv_path = os.path.join(folder_path, "metricas_por_epoca_LSTM_version.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"CSV con métricas guardado en: {csv_path}")

        # Guardar matriz de confusión final por separado
        y_pred_final = modelo.predict([x_test, joints_1_test, joints_2_test])
        y_pred_classes = np.argmax(y_pred_final, axis=1)
        np.savetxt(os.path.join(folder_path, "matriz_confusion_final_LSTM_version.csv"), cm_final, delimiter=",", fmt="%d")        
    
        print("\nResultados finales en el conjunto de prueba:")
        print(f"F1-Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        # Guardar el modelo entrenado con un nombre único
        #modelo_path = get_next_filename(folder_path, "modelo_LSTM_version")
        modelo_path, model_number = get_next_filename(folder_path, "modelo_LSTM_version")
        modelo.save(modelo_path)
        print(f"Modelo guardado en {modelo_path}")
        # Dentro de entrenar_LSTM, después de guardar el modelo:
        guardar_umbrales_en_carpeta(umbrales, folder_path, "umbrales", model_number)  
        
        # Graficar las métricas incluyendo history
        graficar_metricas(metrics_callback, history, folder_path)
        graficar_evolucion_matrices(metrics_callback, folder_path)
        return modelo
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")
        return None

# Guardar los umbrales en un archivo JSON
      
def guardar_umbrales_en_carpeta(umbrales, carpeta_modelo, base_name, model_number):
    try:
        output_base_folder = os.path.join(carpeta_modelo, 'modelo_LSTM_version')
        os.makedirs(output_base_folder, exist_ok=True)  # <-- Crear directorio        
        archivo_umbral = os.path.join(output_base_folder, f"{base_name} {model_number:02d}_umbrales.json")        
        umbrales_convertidos = {
            str(k): [float(v[0]), float(v[1])] 
            for k, v in umbrales.items()
        }        
        with open(archivo_umbral, "w", encoding="utf-8") as json_file:
            json.dump(umbrales_convertidos, json_file, indent=4)
            
    except Exception as e:
        print(f"Error al guardar los umbrales: {e}")
 
def obtener_umbral_por_etiqueta(modelo, x_test, joints_1_test, joints_2_test, y_test):
    print('Empieza a obtener los umbrales')
    
    joints_1_test = np.squeeze(joints_1_test, axis=-1)
    joints_2_test = np.squeeze(joints_2_test, axis=-1)

    prediccion = modelo.predict([x_test, joints_1_test, joints_2_test])
    num_clases = prediccion.shape[1]  # Número de clases según la salida del modelo
    
    umbrales = {}
    for i in range(num_clases):  # Itera sobre todas las clases
        umbrales[i] = (
            float(np.percentile(prediccion[:, i], 5)),  # Convertir a float nativo
            float(np.percentile(prediccion[:, i], 95))
        )
    
    return umbrales



# Programa principal
if __name__ == "__main__":
    carpeta =  'D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet/cantidadMovimiento'
    archivos_csv = buscar_archivos_csv(carpeta)
    if not archivos_csv:
        print("No se encontraron archivos .csv en la carpeta especificada.")
        exit()

    # Crear diccionario directamente desde los resultados
    ruta_archivos = dict(archivos_csv)  # {ruta_csv: etiqueta}

    # Definir las etiquetas esperadas
    ETIQUETAS_VALIDAS = {"En Movimiento", "Parado", "Sentado", "Sin Personas" }
    
    # Filtrar archivos con etiquetas no válidas
    ruta_archivos_filtrado = {
        ruta: etiqueta 
        for ruta, etiqueta in ruta_archivos.items() 
        if etiqueta in ETIQUETAS_VALIDAS }
    
    if not ruta_archivos_filtrado:
        print("No hay archivos con etiquetas válidas.")
        exit()
    
    datos, etiquetas, joint_coords_1, joint_coords_2 = cargar_datos_por_categoria(ruta_archivos)
    folder_path = "D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet/modelos_entrenados"
    if datos is not None:
        x_train, x_test, y_train, y_test = train_test_split(datos, etiquetas, test_size=0.2, random_state=42)
        joints_1_train, joints_1_test = train_test_split(joint_coords_1, test_size=0.2, random_state=42)
        joints_2_train, joints_2_test = train_test_split(joint_coords_2, test_size=0.2, random_state=42)
        
        # Reajustar las formas de los datos
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        joints_1_train = np.expand_dims(joints_1_train, axis=-1)
        joints_1_test = np.expand_dims(joints_1_test, axis=-1)
        joints_2_train = np.expand_dims(joints_2_train, axis=-1)
        joints_2_test = np.expand_dims(joints_2_test, axis=-1)

        # Convertir etiquetas a numéricas si son cadenas
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Extract timesteps and num_features from training data
        timesteps = x_train.shape[1]
        num_features = joints_1_train.shape[1]
        
        # Build and train the model
        modelo = construir_LSTM(timesteps, num_features)
        if modelo:
            print('modelo construido')
            modelo_entrenado = entrenar_LSTM(modelo, x_train, joints_1_train, joints_2_train, y_train, x_test, joints_1_test, joints_2_test, y_test, folder_path)
            print('modelo entrenado finalizado')
            if modelo_entrenado is not None:
                print("empezar el umbral")
                umbrales = obtener_umbral_por_etiqueta(modelo_entrenado, x_test, joints_1_test, joints_2_test, y_test)
                print("Umbrales por etiqueta:", umbrales)               
                guardar_umbrales_en_carpeta(umbrales, folder_path, "umbrales")

"""

"""

import os
import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Importaciones de Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from sklearn.metrics import classification_report, f1_score, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape, Dropout
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Accuracy, Precision, Recall, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import SparseCategoricalAccuracy 
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Flatten, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.callbacks import Callback

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import label_binarize


class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, joints_1_test, joints_2_test, y_test, folder_path, num_classes=4):
        super().__init__()
        self.x_test = x_test
        self.joints_1_test = joints_1_test
        self.joints_2_test = joints_2_test
        self.y_test = y_test
        self.folder_path = folder_path
        self.num_classes = num_classes
        self.epoch_counter = 0
        self.confusion_matrices = []
        
        # Inicializar estructura para almacenar métricas
        self.metrics_history = {
            "epoch": [],
            **{f"precision_{i}": [] for i in range(num_classes)},
            **{f"recall_{i}": [] for i in range(num_classes)},
            **{f"f1_{i}": [] for i in range(num_classes)},
            "accuracy": [],
            "mse": [],
            "mae": [],
            "r2": [],
            "correlation": [],
            "confusion_matrix": []
        }
        
        # Configurar CSV
        self.csv_path = os.path.join(folder_path, "metricas_por_epoca_LSTM.csv")
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.metrics_history.keys()).to_csv(self.csv_path, index=False)        
        # Preparar carpetas para gráficas
        self.preparar_directorios()

    
    def preparar_directorios(self):
        #Crea carpetas para guardar gráficas y matrices de confusión.
        self.reg_plots_dir = os.path.join(self.folder_path, "regression_metrics_plots")
        self.cm_plots_dir = os.path.join(self.folder_path, "confusion_matrices")
        os.makedirs(self.reg_plots_dir, exist_ok=True)
        os.makedirs(self.cm_plots_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_counter += 1
        
        # Paso 1: Predecir con el modelo
        y_pred = self.model.predict(
            [self.x_test, self.joints_1_test, self.joints_2_test], verbose=0 )
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = self.y_test
        
        # Paso 2: Convertir etiquetas a one-hot para métricas de regresión
        y_true_onehot = label_binarize(y_true_classes, classes=np.arange(self.num_classes))
        
        # Paso 3: Calcular métricas globales
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        mse = mean_squared_error(y_true_onehot, y_pred)
        mae = mean_absolute_error(y_true_onehot, y_pred)
        r2 = r2_score(y_true_onehot.flatten(), y_pred.flatten())
        correlation = np.corrcoef(y_true_onehot.flatten(), y_pred.flatten())[0, 1]
        cm = confusion_matrix(y_true_classes, y_pred_classes)
           
        
         # Paso 4: Calcular métricas por clase - FORZAR TODAS LAS CLASES
        report = classification_report(
            y_true_classes,
            y_pred_classes,
            labels=np.arange(self.num_classes),  # <--- ¡Añadir esto!
            target_names=[f"Class_{i}" for i in range(self.num_classes)],
            output_dict=True,
            zero_division=0 )
    
        # Paso 5: Extraer métricas asegurando todas las clases
        class_metrics = {}
        for i in range(self.num_classes):
            class_name = f"Class_{i}"
            class_report = report.get(class_name, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})            
            class_metrics[f"precision_{i}"] = class_report["precision"]
            class_metrics[f"recall_{i}"] = class_report["recall"]
            class_metrics[f"f1_{i}"] = class_report["f1-score"]
    
        # Paso 6: Crear DataFrame con orden de columnas consistente
        df_data = {"epoch": epoch + 1}        
        # Añadir métricas en orden numérico
        for i in range(self.num_classes):
            df_data[f"precision_{i}"] = class_metrics[f"precision_{i}"]
            df_data[f"recall_{i}"] = class_metrics[f"recall_{i}"]
            df_data[f"f1_{i}"] = class_metrics[f"f1_{i}"]
        
        # Añadir métricas globales
        df_data.update({
            "accuracy": accuracy,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "correlation": correlation,
            "confusion_matrix": np.array2string(cm, separator="|") })
    
        # Crear DataFrame con las columnas en orden correcto
        column_order = ["epoch"] + \
                       [f"precision_{i}" for i in range(self.num_classes)] + \
                       [f"recall_{i}" for i in range(self.num_classes)] + \
                       [f"f1_{i}" for i in range(self.num_classes)] + \
                       ["accuracy", "mse", "mae", "r2", "correlation", "confusion_matrix"]
        
        df = pd.DataFrame([df_data], columns=column_order)
    
        # Guardar en CSV
        header = not os.path.exists(self.csv_path)  # Escribir header solo
        
        # Paso 6: Actualizar el historial de métricas
        self.metrics_history["epoch"].append(epoch + 1)
        for metric in class_metrics:
            self.metrics_history[metric].append(class_metrics[metric])
        self.metrics_history["accuracy"].append(accuracy)
        self.metrics_history["mse"].append(mse)
        self.metrics_history["mae"].append(mae)
        self.metrics_history["r2"].append(r2)
        self.metrics_history["correlation"].append(correlation)
        self.metrics_history["confusion_matrix"].append(np.array2string(cm, separator="|"))
        self.confusion_matrices.append(cm)  
        
        # Paso 7: Guardar en CSV
        df_data = {
            "epoch": epoch + 1,
            **class_metrics,
            "accuracy": accuracy,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "correlation": correlation,
            "confusion_matrix": np.array2string(cm, separator="|") }
        df = pd.DataFrame([df_data])
        df.to_csv(self.csv_path, mode="a", header=False, index=False)
        
        # Paso 8: Generar gráficas y matrices
        self.generar_graficas_regresion()
        self.guardar_matriz_confusion(cm, epoch + 1)
        
        # Paso 9: Log en consola
        print(f"\nÉpoca {epoch + 1}:")
        print(f"  - Precisión Global: {accuracy:.4f}")
        print(f"  - F1-Score (Clase 0): {class_metrics['f1_0']:.4f}")

    def generar_graficas_regresion(self):
        #Genera gráficas de evolución de MSE, MAE, R² y Correlación.
        epochs = self.metrics_history["epoch"]
        for metric in ["mse", "mae", "r2", "correlation"]:
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.metrics_history[metric], "b-", label=metric)
            plt.title(f"Evolución de {metric.upper()} por Época")
            plt.xlabel("Época")
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.reg_plots_dir, f"{metric}.png"))
            plt.close()
    
    def guardar_matriz_confusion(self, cm, epoch):
        #Guarda la matriz de confusión como imagen PNG.
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión - Época {epoch}")
        plt.xlabel("Predicciones")
        plt.ylabel("Reales")
        plt.savefig(os.path.join(self.cm_plots_dir, f"cm_epoch_{epoch:03d}.png"))
        plt.close()

     
# Función para buscar archivos CSV
def buscar_archivos_csv(carpeta):
    archivos_csv = []
    for root, dirs, files in os.walk(carpeta):
        for file in files:
            if file.endswith('.csv'):
                ruta_completa = os.path.join(root, file)
                # Obtener la etiqueta desde la carpeta padre directa
                etiqueta = os.path.basename(os.path.dirname(ruta_completa))
                archivos_csv.append((ruta_completa, etiqueta))
    return archivos_csv

 #Función para extraer coordenadas de joints desde archivos TXT

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

def normalizar_joints(joints):
    if joints.shape[0] > 0:
        scaler = MinMaxScaler()
        original_shape = joints.shape
        joints_flat = joints.reshape(-1, joints.shape[-1])
        joints_normalized = scaler.fit_transform(joints_flat)
        return joints_normalized.reshape(original_shape)
    return joints

# Cargar datos y etiquetas
def cargar_datos_por_categoria(ruta_archivos):
    try:
        print('Empieza a cargar datos')
        datos_completos = []
        etiquetas_str = []
        joint_coords_1 = []
        joint_coords_2 = []
        all_coords = []
        
        # 1. Configurar diccionario de etiquetas
        unique_labels = set(ruta_archivos.values())
        labels_dict = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print("Diccionario de etiquetas:", labels_dict)

        # 2. Procesar archivos
        for archivo, etiqueta in ruta_archivos.items():
            if not os.path.exists(archivo):
                continue
                
            datos = pd.read_csv(archivo, encoding='ISO-8859-1')            
            # Validar columnas
            columnas_requeridas = [
                "Cantidad Movimiento", "Promedio Actual", "Promedio Siguiente",
                "Desviacion Actual", "Desviacion Siguiente", 
                "Homogeneidad Actual", "Homogeneidad Siguiente", 
                "Simetria S_J", "Velocidad Promedio"]
            if not all(col in datos.columns for col in columnas_requeridas):
                print(f"¡Archivo {archivo} ignorado: columnas faltantes!")
                continue
            # Añade conversión numérica:
            datos[columnas_requeridas] = datos[columnas_requeridas].apply(pd.to_numeric, errors='coerce')            
            # Luego elimina filas con NaN (si es apropiado):
            datos = datos.dropna(subset=columnas_requeridas)
            
            # Extraer datos principales
            datos_seleccionados = datos[columnas_requeridas].values
            if len(datos_seleccionados) == 0:
                print(f"¡Archivo {archivo} ignorado: sin datos válidos!")
                continue

            # Procesar coordenadas
            joint_paths_1 = datos['Archivo Actual'].values
            joint_paths_2 = datos['Archivo Siguiente'].values
            
            # Asegurar arrays 2D y almacenar
            coords_1 = [_ensure_2d(extract_joints_from_txt(path)) for path in joint_paths_1]
            coords_2 = [_ensure_2d(extract_joints_from_txt(path)) for path in joint_paths_2]
            
            joint_coords_1.extend(coords_1)
            joint_coords_2.extend(coords_2)
            all_coords.extend(joint_coords_1 + joint_coords_2)
            
            datos_completos.append(datos_seleccionados)
            etiquetas_str.extend([etiqueta] * len(datos_seleccionados))

        # 3. Calcular dimensiones máximas
        max_length = max(len(seq) for seq in all_coords)
        max_features = max(np.array(seq).shape[1] for seq in joint_coords_1 + joint_coords_2)

        #max_features = max(seq.shape[1] for seq in joint_coords_1 + joint_coords_2)  # Ahora seguro que son 2D
        print(f"Longitud máxima temporal: {max_length}")
        print(f"Características máximas: {max_features}")

        # 4. Aplicar padding
        joint_coords_1 =[np.pad(seq, ((0, max_length - len(seq)), (0, max_features - seq.shape[1])), mode="constant") for seq in joint_coords_1]      
        joint_coords_2 = [np.pad(seq, ((0, max_length - len(seq)), (0, max_features - seq.shape[1])), mode="constant") for seq in joint_coords_2]
        
        # 5. Convertir a arrays numpy
        joint_coords_1 = np.array(joint_coords_1)
        joint_coords_2 = np.array(joint_coords_2)

        # 6. Normalizar joints (ahora son arrays numpy)
        #joint_coords_1 = normalizar_joints(joint_coords_1)
        #joint_coords_2 = normalizar_joints(joint_coords_2)
        
        # 5. Convertir a arrays numpy
        datos_completos = np.vstack(datos_completos)
        joint_coords_1 = np.array(joint_coords_1)
        joint_coords_2 = np.array(joint_coords_2)
        etiquetas = np.array([labels_dict[label] for label in etiquetas_str], dtype=np.int32)
        
        # Verificación final
        print("\nFormas finales:")
        print(f"- Datos: {datos_completos.shape}")
        print(f"- Joints 1: {joint_coords_1.shape}")
        print(f"- Joints 2: {joint_coords_2.shape}")
        print(f"- Etiquetas: {etiquetas.shape}")        
        return datos_completos, etiquetas, joint_coords_1, joint_coords_2
        
    except Exception as e:
        print(f"¡Error crítico!: {e}")
        return None, None, None, None

# Función auxiliar para asegurar arrays 2D
def _ensure_2d(array):
    if array.ndim == 1:
        return array.reshape(-1, 1)  # Convertir (n,) -> (n, 1)
    return array
       
def get_next_filename(folder_path, base_name):
    output_base_folder = os.path.join(folder_path, 'modelo_LSTM_version')
    os.makedirs(output_base_folder, exist_ok=True)
    
    if not base_name.endswith(" "):
        base_name += " "    
    pattern = re.compile(rf"{re.escape(base_name)}(\d{{2}})_modelo_LSTM.keras")    
    max_num = 0
    for file in os.listdir(output_base_folder):
        match = pattern.match(file)
        if match:
            max_num = max(max_num, int(match.group(1)))    
    next_num = max_num + 1
    new_path = os.path.join(output_base_folder, f"{base_name}{next_num:02d}_modelo_LSTM.keras")    
    # Prevenir colisiones
    while os.path.exists(new_path):
        next_num += 1
        new_path = os.path.join(output_base_folder, f"{base_name}{next_num:02d}_modelo_LSTM.keras")    
    return new_path

#Guardar las metricas en una imagen
def save_metrics_image(folder_path, base_name, figure):
    print('empieza a guardar metricas')
    output_base_folder = os.path.join(folder_path, 'Metricas_imagenes_version') 
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)
        print(f"Se ha creado la carpeta: {output_base_folder}")
    # Filtrar archivos que empiezan con base_name y extraer números correctamente
    existing_files = [f for f in os.listdir(output_base_folder) if f.startswith(base_name)]
    max_num = 0
    for file in existing_files:
        try:
            num_part = file[len(base_name):].strip().split('_')[0]  # Extraer número correctamente
            num = int(num_part)
            max_num = max(max_num, num)
        except (ValueError, IndexError):
            continue  # Si hay un error en la conversión, se ignora el archivo
    next_num = max_num + 1
    image_name = os.path.join(output_base_folder, f"{base_name} {next_num:02d}_modelo_LSTM.png")    
   # Verificar si la figura es válida antes de guardarla
    if isinstance(figure, plt.Figure):
        figure.savefig(image_name)
        plt.close(figure)
        return image_name
    else:
        raise ValueError("El objeto figure no es una instancia válida de matplotlib.figure.Figure")

def graficar_evolucion_matrices(metrics_callback, folder_path):
    plt.figure(figsize=(12, 8))    
    output_base_folder = os.path.join(folder_path, 'Evolucion_cm_epoch_version')
    for i, cm in enumerate(metrics_callback.confusion_matrices):
        plt.clf()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión - Época {i+1}")
        plt.xlabel("Predicciones")
        plt.ylabel("Reales")
        plt.savefig(os.path.join(output_base_folder, f"evolucion_cm_epoch_{i+1:03d}.png"))
        plt.close()
    
def graficar_metricas(metrics_callback, history, image_path):
    print('Empieza a graficar métricas')    
    epochs = range(1, len(metrics_callback.metrics_history["epoch"]) + 1)
    
    # Reducir subplots a 5x2 y ajustar layout
    fig, axs = plt.subplots(5, 2, figsize=(18, 20))
    fig.suptitle('Métricas de Entrenamiento', fontsize=16, y=0.99)
    
    # 1. Gráfico combinado de F1-Score para todas las clases
    num_classes = metrics_callback.num_classes
    colors = plt.cm.viridis(np.linspace(0, 1, num_classes))  # Paleta de colores
    
    axs[0, 0].set_title('F1-Score por Clase')
    for i in range(num_classes):
        axs[0, 0].plot(
            epochs, 
            metrics_callback.metrics_history[f"f1_{i}"], 
            label=f'Clase {i}', 
            color=colors[i],
            linewidth=2,
            alpha=0.8
        )
    axs[0, 0].set_xlabel('Época')
    axs[0, 0].set_ylabel('F1-Score')
    axs[0, 0].legend(loc='lower right', ncol=2)
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy (original)
    axs[0, 1].plot(history.epoch, history.history['accuracy'], label='Entrenamiento', color='navy')
    axs[0, 1].plot(history.epoch, history.history['val_accuracy'], label='Validación', color='skyblue')
    axs[0, 1].set_title('Precisión')
    axs[0, 1].set_xlabel('Época')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Loss (original)
    axs[1, 0].plot(history.epoch, history.history['loss'], label='Entrenamiento', color='darkred')
    axs[1, 0].plot(history.epoch, history.history['val_loss'], label='Validación', color='lightcoral')
    axs[1, 0].set_title('Pérdida')
    axs[1, 0].set_xlabel('Época')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Precisión por clase
    axs[1, 1].set_title('Precisión por Clase')
    for i in range(num_classes):
        axs[1, 1].plot(
            epochs, 
            metrics_callback.metrics_history[f"precision_{i}"], 
            color=colors[i],
            linestyle='--',
            alpha=0.6
        )
    axs[1, 1].set_xlabel('Época')
    axs[1, 1].set_ylabel('Precisión')
    axs[1, 1].grid(True, alpha=0.3)

    # 5. Recall por clase
    axs[2, 0].set_title('Recall por Clase')
    for i in range(num_classes):
        axs[2, 0].plot(
            epochs, 
            metrics_callback.metrics_history[f"recall_{i}"], 
            color=colors[i],
            linestyle='-.',
            alpha=0.6
        )
    axs[2, 0].set_xlabel('Época')
    axs[2, 0].set_ylabel('Recall')
    axs[2, 0].grid(True, alpha=0.3)

    # 6. Métricas de regresión (original)
    metricas_regresion = [
        ('mse', 'MSE', 2, 1),
        ('mae', 'MAE', 3, 0),
        ('r2', 'R²', 3, 1),
        ('correlation', 'Correlación', 4, 0) ]
    
    for metric, name, row, col in metricas_regresion:
        axs[row, col].plot(epochs, metrics_callback.metrics_history[metric], color='purple')
        axs[row, col].set_title(name)
        axs[row, col].set_xlabel('Época')
        axs[row, col].set_ylabel(name)
        axs[row, col].grid(True, alpha=0.3)
    # Eliminar ejes vacíos
    fig.delaxes(axs[4, 1])    
    plt.tight_layout()
    
    # Guardar imagen
    image_path_specific = save_metrics_image(folder_path, "metricas_combinadas", fig)
    image_path_specific = os.path.join(folder_path, 'metricas_combinadas.png')
    plt.savefig(image_path_specific, dpi=300, bbox_inches='tight')
    plt.close()
    return image_path_specific

# Métrica personalizada para Sensibilidad (Recall positivo)
def sensibilidad(y_true, y_pred):
    return tf.keras.metrics.Recall()(y_true, y_pred)

def ajustar_dimensiones(x, expected_timesteps, expected_features):
    x = np.asarray(x, dtype=np.float32)    
    # Eliminar dimensiones de tamaño 1 (excepto batch)
    while len(x.shape) > 3 and x.shape[-1] == 1:
        x = np.squeeze(x, axis=-1)    
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=1)  # Añadir dimensión temporal    
    # Ajustar timesteps y features
    current_timesteps = x.shape[1]
    current_features = x.shape[2]    
    # Padding/Truncado para timesteps
    if current_timesteps < expected_timesteps:
        pad = ((0, 0), (0, expected_timesteps - current_timesteps), (0, 0))
        x = np.pad(x, pad, mode='constant')
    elif current_timesteps > expected_timesteps:
        x = x[:, :expected_timesteps, :]    
    # Padding/Truncado para features
    if current_features < expected_features:
        pad = ((0, 0), (0, 0), (0, expected_features - current_features))
        x = np.pad(x, pad, mode='constant')
    elif current_features > expected_features:
        x = x[:, :, :expected_features]    
    return x

def construir_LSTM(timesteps, num_feature):
    try:
        # Rama para métricas (serie temporal)
        input_metrics = Input(shape=(timesteps, 1), name='metrics_input')
        lstm_metrics = LSTM(64, return_sequences=False)(input_metrics)
        dense_metrics = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(lstm_metrics)  
        
        # Rama para keypoints de la imagen actual
        input_joint1 = Input(shape=(num_feature, 1), name='input_joint1')
        lstm_joint1 = LSTM(64, return_sequences=False)(input_joint1)
        dense_joint1 = Dense(32, activation='relu')(lstm_joint1)
        
        # Rama para keypoints de la imagen siguiente
        input_joint2 = Input(shape=(num_feature, 1), name='input_joint2')
        lstm_joint2 = LSTM(64, return_sequences=False)(input_joint2)
        dense_joint2 = Dense(32, activation='relu')(lstm_joint2)
        
        # Concatenación y fusión
        combinado = Concatenate()([dense_metrics, dense_joint1, dense_joint2])
        combinado = Dense(64, activation='relu')(combinado)
        combinado = Dropout(0.3)(combinado)
        
        # Salida
        salida = Dense(4, activation='softmax')(combinado)        
        modelo = Model(inputs=[input_metrics, input_joint1, input_joint2], outputs=salida)
        modelo.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        modelo.summary()
        return modelo
    except Exception as e:
        print(f"Error al construir el modelo: {e}")
        return None

# Entrenar el modelo LSTM
def entrenar_LSTM(modelo, x_train, joints_1_train, joints_2_train, y_train, 
                  x_test, joints_1_test, joints_2_test, y_test, folder_path):
    try:
       # --- Ajustar dimensiones y eliminar 4D ---
        # Para datos de entrenamiento
        x_train = ajustar_dimensiones(x_train, 9, 1)
        joints_1_train = np.squeeze(joints_1_train, axis=-1)  # <--- Primero squeeze
        joints_1_train = ajustar_dimensiones(joints_1_train, 34, 1)
        joints_2_train = np.squeeze(joints_2_train, axis=-1)  # <--- Primero squeeze
        joints_2_train = ajustar_dimensiones(joints_2_train, 34, 1)
        
        # Para datos de prueba
        x_test = ajustar_dimensiones(x_test, 9, 1)
        joints_1_test = np.squeeze(joints_1_test, axis=-1)    # <--- Primero squeeze
        joints_1_test = ajustar_dimensiones(joints_1_test, 34, 1)
        joints_2_test = np.squeeze(joints_2_test, axis=-1)    # <--- Primero squeeze
        joints_2_test = ajustar_dimensiones(joints_2_test, 34, 1)
        
        print(f"Forma de y_train: {y_train.shape}")
        print(f"Forma de y_test: {y_test.shape}")
        print(f"x_test shape: {x_test.shape}")
        print(f"joints_1_test shape: {joints_1_test.shape}")
        print(f"joints_2_test shape: {joints_2_test.shape}")

        # Crear el callback de métricas
        metrics_callback = MetricsCallback(x_test, joints_1_test, joints_2_test, y_test, folder_path, num_classes=4)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)  
        
        # Entrenar el modelo
        history = modelo.fit(
            [x_train, joints_1_train, joints_2_train], y_train,
            validation_data=([x_test, joints_1_test, joints_2_test], y_test),
            epochs=100, batch_size=32, verbose=2, 
            callbacks=[metrics_callback, early_stop])        
        
        # Evaluar las métricas finales en el conjunto de prueba
        y_pred = modelo.predict([x_test, joints_1_test, joints_2_test])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = y_test  
        print("Etiquetas únicas en y_train:", np.unique(y_train))  

        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        accuracy = accuracy_score(y_true_classes, y_pred_classes) 

        cm = confusion_matrix(y_true_classes, y_pred_classes)   
        cm_final = confusion_matrix(y_test, y_pred_classes)  # ¡Sin np.argmax!  
        # Guardar todas las métricas en un CSV
        df_metrics = pd.DataFrame(metrics_callback.metrics_history)
        csv_path = os.path.join(folder_path, "metricas_por_epoca_LSTM_version.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"CSV con métricas guardado en: {csv_path}")

        # Guardar matriz de confusión final por separado
        y_pred_final = modelo.predict([x_test, joints_1_test, joints_2_test])
        y_pred_classes = np.argmax(y_pred_final, axis=1)
        np.savetxt(os.path.join(folder_path, "matriz_confusion_final_LSTM_version.csv"), cm_final, delimiter=",", fmt="%d")        
    
        print("\nResultados finales en el conjunto de prueba:")
        print(f"F1-Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        # Guardar el modelo entrenado con un nombre único
        modelo_path = get_next_filename(folder_path, "modelo_LSTM")
        modelo.save(modelo_path)
        print(f"Modelo guardado en {modelo_path}")
        
        # Graficar las métricas incluyendo history
        graficar_metricas(metrics_callback, history, folder_path)
        graficar_evolucion_matrices(metrics_callback, folder_path)
        return modelo
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")
        return None

# Guardar los umbrales en un archivo JSON
def guardar_umbrales_en_carpeta(umbrales, carpeta_modelo, base_name):
    try:
        output_base_folder = os.path.join(carpeta_modelo, 'modelo_LSTM_version')
        os.makedirs(output_base_folder, exist_ok=True)
        
        existing_files = [f for f in os.listdir(output_base_folder) if f.startswith(base_name)]
        max_num = 0
        for f in existing_files:
            match = re.findall(r"(\d+)_umbrales\.json$", f)
            if match:
                max_num = max(max_num, int(match[0]))
        archivo_umbral = os.path.join(output_base_folder, f"{base_name} {max_num + 1:02d}_umbrales.json")
        # Convertir numpy.float32 a float y claves a strings
        umbrales_convertidos = {
            str(k): [float(v[0]), float(v[1])]  # Asegurar conversión a float
            for k, v in umbrales.items()}        
        with open(archivo_umbral, "w", encoding="utf-8") as json_file:
            json.dump(umbrales_convertidos, json_file, indent=4)
            
        print(f"Archivo guardado correctamente en: {archivo_umbral}")
    except Exception as e:
        print(f"Error al guardar los umbrales: {e}")
 
def obtener_umbral_por_etiqueta(modelo, x_test, joints_1_test, joints_2_test, y_test):
    print('empieza a obtener los umbrales')
    
    # Depurar dimensiones
    joints_1_test = np.squeeze(joints_1_test, axis=-1)
    joints_2_test = np.squeeze(joints_2_test, axis=-1)

    # Realizar predicciones
    prediccion = modelo.predict([x_test, joints_1_test, joints_2_test])
    umbrales = {}
    for i, etiqueta in enumerate(np.unique(y_test)):
        umbrales[etiqueta] = (np.percentile(prediccion[:, i], 5), np.percentile(prediccion[:, i], 95))
    return umbrales



# Programa principal
if __name__ == "__main__":
    carpeta =  'D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet/cantidadMovimiento'
    archivos_csv = buscar_archivos_csv(carpeta)
    if not archivos_csv:
        print("No se encontraron archivos .csv en la carpeta especificada.")
        exit()

    # Crear diccionario directamente desde los resultados
    ruta_archivos = dict(archivos_csv)  # {ruta_csv: etiqueta}

    # Definir las etiquetas esperadas
    ETIQUETAS_VALIDAS = {"En Movimiento", "Parado", "Sentado", "Sin Personas" }
    
    # Filtrar archivos con etiquetas no válidas
    ruta_archivos_filtrado = {
        ruta: etiqueta 
        for ruta, etiqueta in ruta_archivos.items() 
        if etiqueta in ETIQUETAS_VALIDAS }
    
    if not ruta_archivos_filtrado:
        print("No hay archivos con etiquetas válidas.")
        exit()
    
    datos, etiquetas, joint_coords_1, joint_coords_2 = cargar_datos_por_categoria(ruta_archivos)
    folder_path = "D:/programs/redNeuronal/modificaciones/datos agrupar/modelos_entrenados"
    if datos is not None:
        x_train, x_test, y_train, y_test = train_test_split(datos, etiquetas, test_size=0.2, random_state=42)
        joints_1_train, joints_1_test = train_test_split(joint_coords_1, test_size=0.2, random_state=42)
        joints_2_train, joints_2_test = train_test_split(joint_coords_2, test_size=0.2, random_state=42)
        
        # Reajustar las formas de los datos
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        joints_1_train = np.expand_dims(joints_1_train, axis=-1)
        joints_1_test = np.expand_dims(joints_1_test, axis=-1)
        joints_2_train = np.expand_dims(joints_2_train, axis=-1)
        joints_2_test = np.expand_dims(joints_2_test, axis=-1)

        # Convertir etiquetas a numéricas si son cadenas
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Extract timesteps and num_features from training data
        timesteps = x_train.shape[1]
        num_features = joints_1_train.shape[1]
        
        # Build and train the model
        modelo = construir_LSTM(timesteps, num_features)
        if modelo:
            print('modelo construido')
            modelo_entrenado = entrenar_LSTM(modelo, x_train, joints_1_train, joints_2_train, y_train, x_test, joints_1_test, joints_2_test, y_test, folder_path)
            print('modelo entrenado finalizado')
            if modelo_entrenado is not None:
                print("empezar el umbral")
                umbrales = obtener_umbral_por_etiqueta(modelo_entrenado, x_test, joints_1_test, joints_2_test, y_test)
                print("Umbrales por etiqueta:", umbrales)               
                guardar_umbrales_en_carpeta(umbrales, folder_path, "umbrales")"""



import os
import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Importaciones de Keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Reshape, Dropout
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout, Attention
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Accuracy, Precision, Recall, MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.metrics import SparseCategoricalAccuracy 
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate, Flatten, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.callbacks import Callback

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import label_binarize


class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_test, joints_1_test, joints_2_test, y_test, folder_path, num_classes=4):
        super().__init__()
        self.x_test = x_test
        self.joints_1_test = joints_1_test
        self.joints_2_test = joints_2_test
        self.y_test = y_test
        self.folder_path = folder_path
        self.num_classes = num_classes
        self.epoch_counter = 0
        self.confusion_matrices = []
        
        # Inicializar estructura para almacenar métricas
        self.metrics_history = {
            "epoch": [],
            **{f"precision_{i}": [] for i in range(num_classes)},
            **{f"recall_{i}": [] for i in range(num_classes)},
            **{f"f1_{i}": [] for i in range(num_classes)},
            "accuracy": [],
            "f1_score": [],
            "recall": [],
            "mse": [],
            "mae": [],
            "r2": [],
            "correlation": [],
            "confusion_matrix": []
        }
        
        # Configurar CSV
        self.csv_path = os.path.join(folder_path, "metricas_por_epoca_LSTM.csv")
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.metrics_history.keys()).to_csv(self.csv_path, index=False)        
        # Preparar carpetas para gráficas
        self.preparar_directorios()
        
    def preparar_directorios(self):
        """Crea carpetas para guardar gráficas y matrices de confusión."""
        self.reg_plots_dir = os.path.join(self.folder_path, "regression_metrics_plots")
        self.cm_plots_dir = os.path.join(self.folder_path, "confusion_matrices")
        os.makedirs(self.reg_plots_dir, exist_ok=True)
        os.makedirs(self.cm_plots_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_counter += 1
        
        # Paso 1: Predecir con el modelo
        y_pred = self.model.predict(
            [self.x_test, self.joints_1_test, self.joints_2_test], verbose=0 )
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = self.y_test
        
        # Paso 2: Convertir etiquetas a one-hot para métricas de regresión
        y_true_onehot = label_binarize(y_true_classes, classes=np.arange(self.num_classes))
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        # Paso 3: Calcular métricas globales
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        mse = mean_squared_error(y_true_onehot, y_pred)
        mae = mean_absolute_error(y_true_onehot, y_pred)
        r2 = r2_score(y_true_onehot.flatten(), y_pred.flatten())
        correlation = np.corrcoef(y_true_onehot.flatten(), y_pred.flatten())[0, 1]
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        # Calcular matriz de confusión
        
         # Paso 4: Calcular métricas por clase - FORZAR TODAS LAS CLASES
        report = classification_report(
            y_true_classes,
            y_pred_classes,
            labels=np.arange(self.num_classes),  # <--- ¡Añadir esto!
            target_names=[f"Class_{i}" for i in range(self.num_classes)],
            output_dict=True,
            zero_division=0 )        
    
        # Paso 5: Extraer métricas asegurando todas las clases
        class_metrics = {}
        for i in range(self.num_classes):
            class_name = f"Class_{i}"
            class_report = report.get(class_name, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0})            
            class_metrics[f"precision_{i}"] = class_report["precision"]
            class_metrics[f"recall_{i}"] = class_report["recall"]
            class_metrics[f"f1_{i}"] = class_report["f1-score"]
    
        # Paso 6: Crear DataFrame con orden de columnas consistente
        df_data = {"epoch": epoch + 1}        
        # Añadir métricas en orden numérico
        for i in range(self.num_classes):
            df_data[f"precision_{i}"] = class_metrics[f"precision_{i}"]
            df_data[f"recall_{i}"] = class_metrics[f"recall_{i}"]
            df_data[f"f1_{i}"] = class_metrics[f"f1_{i}"]
        
        # Añadir métricas globales
        df_data.update({
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "correlation": correlation,
            "confusion_matrix": np.array2string(cm, separator="|")
            })
        
        # Crear DataFrame con las columnas en orden correcto
        column_order = ["epoch"] + \
                    [f"precision_{i}" for i in range(self.num_classes)] + \
                    [f"recall_{i}" for i in range(self.num_classes)] + \
                    [f"f1_{i}" for i in range(self.num_classes)] + \
                    ["f1-score", "recall", "accuracy", "mse", "mae", "r2", "correlation", "confusion_matrix"]
        df = pd.DataFrame([df_data], columns=column_order)
    
        # Guardar en CSV
        header = not os.path.exists(self.csv_path)  # Escribir header solo
        
        # Paso 6: Actualizar el historial de métricas
        self.metrics_history["epoch"].append(epoch + 1)
        for metric in class_metrics:
            self.metrics_history[metric].append(class_metrics[metric])
        self.metrics_history["accuracy"].append(accuracy)
        self.metrics_history["mse"].append(mse)
        self.metrics_history["mae"].append(mae)
        # Actualizar metrics_history:
        self.metrics_history["f1_score"].append(f1)
        self.metrics_history["recall"].append(recall)

        self.metrics_history["r2"].append(r2)
        self.metrics_history["correlation"].append(correlation)
        self.metrics_history["confusion_matrix"].append(np.array2string(cm, separator="|"))
        self.confusion_matrices.append(cm)  
        
        # Paso 7: Guardar en CSV
        df_data = {
            "epoch": epoch + 1,
            **class_metrics,
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall,
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "correlation": correlation,
            "confusion_matrix": np.array2string(cm, separator="|") }
        df = pd.DataFrame([df_data])
        df.to_csv(self.csv_path, mode="a", header=False, index=False)
        
        # Paso 8: Generar gráficas y matrices
        self.generar_graficas_regresion()
        self.guardar_matriz_confusion(cm, epoch + 1)      
        # Guardar matriz final
        self.guardar_matriz_confusion_final(cm)        
        # Opcional: Guardar también en formato CSV
        np.savetxt(os.path.join(self.folder_path, "confusion_matrix_final.csv"), 
                cm, fmt="%d", delimiter=",")
        
        # Paso 9: Log en consola
        print(f"\nÉpoca {epoch + 1}:")
        print(f"  - Precisión Global: {accuracy:.4f}")
        print(f"  - F1-Score (Clase 0): {class_metrics['f1_0']:.4f}")

    def generar_graficas_regresion(self):
        """Genera gráficas de evolución de MSE, MAE, R² y Correlación."""
        epochs = self.metrics_history["epoch"]
        for metric in ["mse", "mae", "r2", "correlation"]:
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, self.metrics_history[metric], "b-", label=metric)
            plt.title(f"Evolución de {metric.upper()} por Época")
            plt.xlabel("Época")
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.reg_plots_dir, f"{metric}.png"))
            plt.close()
    
    def guardar_matriz_confusion(self, cm, epoch):
        """Guarda la matriz de confusión como imagen PNG."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusión - Época {epoch}")
        plt.xlabel("Predicciones")
        plt.ylabel("Reales")
        plt.savefig(os.path.join(self.cm_plots_dir, f"cm_epoch_{epoch:03d}.png"))
        plt.close()
    
    def guardar_matriz_confusion_final(self, cm):
        """Guarda la matriz de confusión final del conjunto de prueba."""
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    annot_kws={"size": 12}, cbar=False)
        plt.title("Matriz de Confusión - Conjunto de Prueba Final", fontsize=14)
        plt.xlabel("Predicciones", fontsize=12)
        plt.ylabel("Valores Reales", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
        # Guardar en alta resolución
        plt.savefig(os.path.join(self.folder_path, "confusion_matrix_final.png"), 
                    dpi=300, bbox_inches='tight')
        plt.close()
     
# Función para buscar archivos CSV
def buscar_archivos_csv(carpeta):
    archivos_csv = []
    for root, dirs, files in os.walk(carpeta):
        for file in files:
            if file.endswith('.csv'):
                ruta_completa = os.path.join(root, file)
                # Obtener la etiqueta desde la carpeta padre directa
                etiqueta = os.path.basename(os.path.dirname(ruta_completa))
                archivos_csv.append((ruta_completa, etiqueta))
    return archivos_csv

 #Función para extraer coordenadas de joints desde archivos TXT

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

def normalizar_joints(joints):
    if joints.shape[0] > 0:
        scaler = MinMaxScaler()
        original_shape = joints.shape
        joints_flat = joints.reshape(-1, joints.shape[-1])
        joints_normalized = scaler.fit_transform(joints_flat)
        return joints_normalized.reshape(original_shape)
    return joints

# Cargar datos y etiquetas
def cargar_datos_por_categoria(ruta_archivos):
    try:
        print('Empieza a cargar datos')
        datos_completos = []
        etiquetas_str = []
        joint_coords_1 = []
        joint_coords_2 = []
        all_coords = []
        
        # 1. Configurar diccionario de etiquetas
        unique_labels = set(ruta_archivos.values())
        labels_dict = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        print("Diccionario de etiquetas:", labels_dict)

        # 2. Procesar archivos
        for archivo, etiqueta in ruta_archivos.items():
            if not os.path.exists(archivo):
                continue
                
            datos = pd.read_csv(archivo, encoding='ISO-8859-1')            
            # Validar columnas
            columnas_requeridas = [
                "Cantidad Movimiento", "Promedio Actual", "Promedio Siguiente",
                "Desviacion Actual", "Desviacion Siguiente", 
                "Homogeneidad Actual", "Homogeneidad Siguiente", 
                "Simetria S_J", "Velocidad Promedio"]
            if not all(col in datos.columns for col in columnas_requeridas):
                print(f"¡Archivo {archivo} ignorado: columnas faltantes!")
                continue
            # Añade conversión numérica:
            datos[columnas_requeridas] = datos[columnas_requeridas].apply(pd.to_numeric, errors='coerce')            
            # Luego elimina filas con NaN (si es apropiado):
            datos = datos.dropna(subset=columnas_requeridas)
            
            # Extraer datos principales
            datos_seleccionados = datos[columnas_requeridas].values
            if len(datos_seleccionados) == 0:
                print(f"¡Archivo {archivo} ignorado: sin datos válidos!")
                continue

            # Procesar coordenadas
            joint_paths_1 = datos['Archivo Actual'].values
            joint_paths_2 = datos['Archivo Siguiente'].values
            
            # Asegurar arrays 2D y almacenar
            coords_1 = [_ensure_2d(extract_joints_from_txt(path)) for path in joint_paths_1]
            coords_2 = [_ensure_2d(extract_joints_from_txt(path)) for path in joint_paths_2]
            
            joint_coords_1.extend(coords_1)
            joint_coords_2.extend(coords_2)
            all_coords.extend(joint_coords_1 + joint_coords_2)
            
            datos_completos.append(datos_seleccionados)
            etiquetas_str.extend([etiqueta] * len(datos_seleccionados))

        # 3. Calcular dimensiones máximas
        max_length = max(len(seq) for seq in all_coords)
        max_features = max(np.array(seq).shape[1] for seq in joint_coords_1 + joint_coords_2)

        #max_features = max(seq.shape[1] for seq in joint_coords_1 + joint_coords_2)  # Ahora seguro que son 2D
        print(f"Longitud máxima temporal: {max_length}")
        print(f"Características máximas: {max_features}")

        # 4. Aplicar padding
        joint_coords_1 =[np.pad(seq, ((0, max_length - len(seq)), (0, max_features - seq.shape[1])), mode="constant") for seq in joint_coords_1]      
        joint_coords_2 = [np.pad(seq, ((0, max_length - len(seq)), (0, max_features - seq.shape[1])), mode="constant") for seq in joint_coords_2]
        
        # 5. Convertir a arrays numpy
        joint_coords_1 = np.array(joint_coords_1)
        joint_coords_2 = np.array(joint_coords_2)

        # 5. Convertir a arrays numpy
        datos_completos = np.vstack(datos_completos)
        joint_coords_1 = np.array(joint_coords_1)
        joint_coords_2 = np.array(joint_coords_2)
        etiquetas = np.array([labels_dict[label] for label in etiquetas_str], dtype=np.int32)
        
        # Verificación final
        print("\nFormas finales:")
        print(f"- Datos: {datos_completos.shape}")
        print(f"- Joints 1: {joint_coords_1.shape}")
        print(f"- Joints 2: {joint_coords_2.shape}")
        print(f"- Etiquetas: {etiquetas.shape}")        
        return datos_completos, etiquetas, joint_coords_1, joint_coords_2
        
    except Exception as e:
        print(f"¡Error crítico!: {e}")
        return None, None, None, None

# Función auxiliar para asegurar arrays 2D
def _ensure_2d(array):
    if array.ndim == 1:
        return array.reshape(-1, 1)  # Convertir (n,) -> (n, 1)
    return array
       
def get_next_filename(folder_path, base_name):
    output_base_folder = os.path.join(folder_path, 'modelo_LSTM_version')
    os.makedirs(output_base_folder, exist_ok=True)
    
    base_name = base_name.strip() + " "  # Asegurar espacio al final
    pattern = re.compile(rf"{re.escape(base_name)}(\d{{2}})_modelo_LSTM\.keras")  # Escapar puntos
    
    max_num = 0
    for file in os.listdir(output_base_folder):
        match = pattern.match(file)
        if match:
            max_num = max(max_num, int(match.group(1)))
    
    next_num = max_num + 1
    new_path = os.path.join(output_base_folder, f"{base_name}{next_num:02d}_modelo_LSTM.keras")
    
    # Prevenir colisiones (por si acaso)
    while os.path.exists(new_path):
        next_num += 1
        new_path = os.path.join(output_base_folder, f"{base_name}{next_num:02d}_modelo_LSTM.keras")
    
    return new_path, next_num

#Guardar las metricas en una imagen
def save_metrics_image(folder_path, base_name, figure):
    print('Empieza a guardar metricas')
    output_base_folder = os.path.join(folder_path, 'metricas_Otras')
    os.makedirs(output_base_folder, exist_ok=True)  # Solo crea la carpeta principal
    
    # Buscar archivos existentes y determinar el próximo número
    existing_files = [f for f in os.listdir(output_base_folder) 
                     if f.startswith(base_name) and f.endswith(".png")]
    max_num = 0
    
    # Regex mejorado para extraer números
    pattern = re.compile(rf"{re.escape(base_name)}\s*(\d+)_modelo_LSTM\.png")
    for file in existing_files:
        match = pattern.match(file)
        if match:
            file_num = int(match.group(1))
            max_num = max(max_num, file_num)
    
    next_num = max_num + 1
    image_name = os.path.join(output_base_folder, f"{base_name} {next_num}_modelo_LSTM.png")    
    # Guardar figura sin crear directorios adicionales
    figure.savefig(image_name, dpi=300, bbox_inches='tight')
    plt.close(figure)    
    return image_name
    
    
def graficar_metricas(metrics_callback, history, folder_path):
    print('Empieza a graficar métricas')
    print(f"Longitud de métricas: {len(metrics_callback.metrics_history['epoch'])}")
    print(f"Longitud de history: {len(history.epoch)}")
    
    # Verificar si hay datos para graficar
    if not metrics_callback.metrics_history.get("epoch") or not history.epoch:
        print("Error: No hay datos de métricas o historial para graficar")
        return None
    
    epochs = range(1, len(metrics_callback.metrics_history["epoch"]) + 1)
    # Configurar figura única
    fig = plt.figure(figsize=(25, 30))
    gs = fig.add_gridspec(6, 2, height_ratios=[1.5, 1, 1.5, 1, 1.5, 1])
    
    # Subplots principales
    ax0 = fig.add_subplot(gs[0, 0])  # F1 por clase
    ax1 = fig.add_subplot(gs[0, 1])  # Macro F1 
    ax2 = fig.add_subplot(gs[1, 0])  # Accuracy
    ax3 = fig.add_subplot(gs[1, 1])  # Precisión por clase
    ax4 = fig.add_subplot(gs[2, 0])  # Recall por clase
    ax5 = fig.add_subplot(gs[2, 1])  # Macro Recall 
    ax6 = fig.add_subplot(gs[3, 0])  # Loss
    ax7 = fig.add_subplot(gs[3, 1])  # MSE
    ax8 = fig.add_subplot(gs[4, 0])  # MAE
    ax9 = fig.add_subplot(gs[4, 1])  # R²
    ax10 = fig.add_subplot(gs[5, 0]) # Correlación
    
    # 1. F1 por clase
    colors = plt.cm.tab20(np.linspace(0, 1, metrics_callback.num_classes))
    for i in range(metrics_callback.num_classes):
        ax0.plot(epochs, metrics_callback.metrics_history[f"f1_{i}"], 
                label=f'Clase {i}', color=colors[i], linewidth=1.5)
    ax0.set_title('Evolución de F1-Score por Clase', fontsize=12)
    ax0.legend(ncol=2, fontsize=8)
    ax0.grid(alpha=0.2)

    # 2. Macro F1
    ax1.plot(epochs, metrics_callback.metrics_history["f1_score"], 'green', linewidth=2, label='F1 Global')
    ax1.set_title('F1-Score Global', fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(alpha=0.2)

    # 3. Precisión por clase
    for i in range(metrics_callback.num_classes):
        ax2.plot(epochs, metrics_callback.metrics_history[f"precision_{i}"], 
                color=colors[i], linestyle='--', alpha=0.7, label=f'Clase {i}')
    ax2.set_title('Precisión por Clase', fontsize=12)
    ax2.legend(ncol=2, fontsize=8)
    ax2.grid(alpha=0.2)

    # 4. Accuracy
    ax3.plot(history.epoch, history.history['accuracy'], 'navy', label='Entrenamiento')
    ax3.plot(history.epoch, history.history['val_accuracy'], 'skyblue', label='Validación')
    ax3.set_title('Precisión Global', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.2)

    # 5. Recall por clase
    for i in range(metrics_callback.num_classes):
        ax4.plot(epochs, metrics_callback.metrics_history[f"recall_{i}"], 
                color=colors[i], linestyle='-.', alpha=0.7, label=f'Clase {i}')
    ax4.set_title('Recall por Clase', fontsize=12)
    ax4.legend(ncol=2, fontsize=8)
    ax4.grid(alpha=0.2)
    
    # 6. Macro Recall
    ax5.plot(epochs, metrics_callback.metrics_history["recall"], 'blue', linewidth=2, label='Recall Global')
    ax5.set_title('Recall Global', fontsize=10)
    ax5.set_ylim(0, 1)
    ax5.legend()
    ax5.grid(alpha=0.2)

    # 7. Loss
    ax6.plot(history.epoch, history.history['loss'], 'darkred', label='Entrenamiento')
    ax6.plot(history.epoch, history.history['val_loss'], 'lightcoral', label='Validación')
    ax6.set_title('Pérdida', fontsize=12)
    ax6.legend()
    ax6.grid(alpha=0.2) 

    # 8. Métricas de regresión
    ax7.plot(epochs, metrics_callback.metrics_history["mse"], 'purple', label='MSE')
    ax7.set_title('MSE', fontsize=12)
    ax7.legend()
    ax7.grid(alpha=0.2)  
    
    ax8.plot(epochs, metrics_callback.metrics_history["mae"], 'orange', label='MAE')
    ax8.set_title('MAE', fontsize=12)
    ax8.legend()
    ax8.grid(alpha=0.2)
    
    ax9.plot(epochs, metrics_callback.metrics_history["r2"], 'red', label='R²')
    ax9.set_title('R²', fontsize=12)
    ax9.legend()
    ax9.grid(alpha=0.2)
    
    ax10.plot(epochs, metrics_callback.metrics_history["correlation"], 'magenta', label='Correlación')
    ax10.set_title('Correlación', fontsize=12)
    ax10.legend()
    ax10.grid(alpha=0.2)
    plt.tight_layout()
    
    # Guardar usando la función de guardado
    image_path = save_metrics_image(folder_path, "metricas_Otras", fig)
    print(f"Gráfica de métricas guardada en: {image_path}")
    return image_path
       

# Métrica personalizada para Sensibilidad (Recall positivo)
def sensibilidad(y_true, y_pred):
    return tf.keras.metrics.Recall()(y_true, y_pred)

def ajustar_dimensiones(x, expected_timesteps, expected_features):
    x = np.asarray(x, dtype=np.float32)    
    # Eliminar dimensiones de tamaño 1 (excepto batch)
    while len(x.shape) > 3 and x.shape[-1] == 1:
        x = np.squeeze(x, axis=-1)    
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=1)  # Añadir dimensión temporal    
    # Ajustar timesteps y features
    current_timesteps = x.shape[1]
    current_features = x.shape[2]    
    # Padding/Truncado para timesteps
    if current_timesteps < expected_timesteps:
        pad = ((0, 0), (0, expected_timesteps - current_timesteps), (0, 0))
        x = np.pad(x, pad, mode='constant')
    elif current_timesteps > expected_timesteps:
        x = x[:, :expected_timesteps, :]    
    # Padding/Truncado para features
    if current_features < expected_features:
        pad = ((0, 0), (0, 0), (0, expected_features - current_features))
        x = np.pad(x, pad, mode='constant')
    elif current_features > expected_features:
        x = x[:, :, :expected_features]    
    return x

def construir_LSTM(timesteps=30, num_feature=30):
    try:
        # Rama para métricas (serie temporal)
        input_metrics = Input(shape=( timesteps, 9), name='metrics_input')
        lstm_metrics = LSTM(128, return_sequences=False)(input_metrics)
        dense_metrics = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(lstm_metrics)  
        
        # Rama para keypoints de la imagen actual
        input_joint1 = Input(shape=( num_feature, 17), name='input_joint1')
        lstm_joint1 = LSTM(128, return_sequences=False)(input_joint1)
        dense_joint1 = Dense(64, activation='relu')(lstm_joint1)
        
        # Rama para keypoints de la imagen siguiente
        input_joint2 = Input(shape=(num_feature, 17), name='input_joint2')
        lstm_joint2 = LSTM(128, return_sequences=False)(input_joint2)
        dense_joint2 = Dense(64, activation='relu')(lstm_joint2)
        
        # Concatenación y fusión
        combinado = Concatenate()([dense_metrics, dense_joint1, dense_joint2])
        combinado = Dense(64, activation='relu')(combinado)
        combinado = Dropout(0.3)(combinado)
        
        # Salida
        salida = Dense(4, activation='softmax')(combinado)        
        modelo = Model(inputs=[input_metrics, input_joint1, input_joint2], outputs=salida)
        modelo.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        modelo.summary()
        return modelo
    except Exception as e:
        print(f"Error al construir el modelo: {e}")
        return None

# Entrenar el modelo LSTM
def entrenar_LSTM(modelo, x_train, joints_1_train, joints_2_train, y_train, 
                  x_test, joints_1_test, joints_2_test, y_test, folder_path):
    try:
       # --- Ajustar dimensiones y eliminar 4D ---
        # Para datos de entrenamiento
        x_train = ajustar_dimensiones(x_train, 9, 9)
        #joints_1_train = np.squeeze(joints_1_train, axis=-1)  # <--- Primero squeeze
        joints_1_train = ajustar_dimensiones(joints_1_train, 34, 17)
        #joints_2_train = np.squeeze(joints_2_train, axis=-1)  # <--- Primero squeeze
        joints_2_train = ajustar_dimensiones(joints_2_train, 34, 17)
        
        # Para datos de prueba
        x_test = ajustar_dimensiones(x_test, 9, 9)
        #joints_1_test = np.squeeze(joints_1_test, axis=-1)    # <--- Primero squeeze
        joints_1_test = ajustar_dimensiones(joints_1_test, 34, 17)
        #joints_2_test = np.squeeze(joints_2_test, axis=-1)    # <--- Primero squeeze
        joints_2_test = ajustar_dimensiones(joints_2_test, 34, 17)
        
        print(f"Forma de y_train: {y_train.shape}")
        print(f"Forma de y_test: {y_test.shape}")
        print(f"x_test shape: {x_test.shape}")
        print(f"joints_1_test shape: {joints_1_test.shape}")
        print(f"joints_2_test shape: {joints_2_test.shape}")

        # Crear el callback de métricas
        metrics_callback = MetricsCallback(x_test, joints_1_test, joints_2_test, y_test, folder_path, num_classes=4)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)  
        
        # Entrenar el modelo
        history = modelo.fit(
            [x_train, joints_1_train, joints_2_train], y_train,
            validation_data=([x_test, joints_1_test, joints_2_test], y_test),
            epochs=100, batch_size=32, verbose=2, 
            callbacks=[metrics_callback, early_stop])        
        
        # Evaluar las métricas finales en el conjunto de prueba
        y_pred = modelo.predict([x_test, joints_1_test, joints_2_test])
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = y_test  
        print("Etiquetas únicas en y_train:", np.unique(y_train))  

        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        accuracy = accuracy_score(y_true_classes, y_pred_classes) 

        cm = confusion_matrix(y_true_classes, y_pred_classes)   
        cm_final = confusion_matrix(y_test, y_pred_classes)  
        # Guardar todas las métricas en un CSV
        df_metrics = pd.DataFrame(metrics_callback.metrics_history)
        csv_path = os.path.join(folder_path, "metricas_por_epoca_LSTM_version.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"CSV con métricas guardado en: {csv_path}")

        # Guardar matriz de confusión final por separado
        y_pred_final = modelo.predict([x_test, joints_1_test, joints_2_test])
        y_pred_classes = np.argmax(y_pred_final, axis=1)
        np.savetxt(os.path.join(folder_path, "matriz_confusion_final_LSTM_version.csv"), cm_final, delimiter=",", fmt="%d")        
    
        print("\nResultados finales en el conjunto de prueba:")
        print(f"F1-Score: {f1:.10f}")
        print(f"Recall: {recall:.10f}")
        print(f"Precision: {precision:.10f}")
        print(f"Accuracy: {accuracy:.10f}")
        print(f"Confusion Matrix:\n{cm}")
        # Llamar así la función
        graficar_metricas(metrics_callback, history, folder_path)
        umbrales = obtener_umbral_por_etiqueta(modelo, x_train, joints_1_train, joints_2_train, y_train)        
        #umbrales = obtener_umbral_por_etiqueta(modelo, x_test, joints_1_test, joints_2_test, y_test)
        #validar_umbrales(umbrales, y_true_classes, y_pred_classes)
        # Guardar el modelo entrenado con un nombre único
        modelo_path, model_number = get_next_filename(folder_path, "modelo_LSTM_version")        
        modelo.save(modelo_path)
        print(f"Modelo guardado en {modelo_path}")
        guardar_umbrales_en_carpeta(umbrales, folder_path, "umbrales", model_number)
       
        #graficar_evolucion_matrices(metrics_callback, folder_path)
        return modelo
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")
        return None# Guardar los umbrales en un archivo JSON
def guardar_umbrales_en_carpeta(umbrales, carpeta_modelo, base_name):
    try:
        output_base_folder = os.path.join(carpeta_modelo, 'modelo_LSTM_version')
        os.makedirs(output_base_folder, exist_ok=True)
        
        existing_files = [f for f in os.listdir(output_base_folder) if f.startswith(base_name)]
        max_num = 0
        for f in existing_files:
            match = re.findall(r"(\d+)_umbrales\.json$", f)
            if match:
                max_num = max(max_num, int(match[0]))
        archivo_umbral = os.path.join(output_base_folder, f"{base_name} {max_num + 1:02d}_umbrales.json")
        # Convertir numpy.float32 a float y claves a strings
        umbrales_convertidos = {
            str(k): [float(v[0]), float(v[1])]  # Asegurar conversión a float
            for k, v in umbrales.items()}        
        with open(archivo_umbral, "w", encoding="utf-8") as json_file:
            json.dump(umbrales_convertidos, json_file, indent=4)
            
        print(f"Archivo guardado correctamente en: {archivo_umbral}")
    except Exception as e:
        print(f"Error al guardar los umbrales: {e}")
 
def obtener_umbral_por_etiqueta(modelo, x_test, joints_1_test, joints_2_test, y_test):
    print('empieza a obtener los umbrales')
    
    # Depurar dimensiones
    joints_1_test = np.squeeze(joints_1_test, axis=-1)
    joints_2_test = np.squeeze(joints_2_test, axis=-1)

    # Realizar predicciones
    prediccion = modelo.predict([x_test, joints_1_test, joints_2_test])
    umbrales = {}
    for i, etiqueta in enumerate(np.unique(y_test)):
        umbrales[etiqueta] = (np.percentile(prediccion[:, i], 5), np.percentile(prediccion[:, i], 95))
    return umbrales


# Programa principal
if __name__ == "__main__":    
    import time
    start_time = time.time()  
    
    carpeta =  'D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet/cantidadMovimiento'
    archivos_csv = buscar_archivos_csv(carpeta)
    if not archivos_csv:
        print("No se encontraron archivos .csv en la carpeta especificada.")
        exit()

    # Crear diccionario directamente desde los resultados
    ruta_archivos = dict(archivos_csv)  # {ruta_csv: etiqueta}

    # Definir las etiquetas esperadas
    ETIQUETAS_VALIDAS = {"En Movimiento", "Parado", "Sentado", "Sin Personas" }
    
    # Filtrar archivos con etiquetas no válidas
    ruta_archivos_filtrado = {
        ruta: etiqueta 
        for ruta, etiqueta in ruta_archivos.items() 
        if etiqueta in ETIQUETAS_VALIDAS }
    
    if not ruta_archivos_filtrado:
        print("No hay archivos con etiquetas válidas.")
        exit()
    
    datos, etiquetas, joint_coords_1, joint_coords_2 = cargar_datos_por_categoria(ruta_archivos)
    folder_path = "D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet/modelos_entrenados"
    if datos is not None:
        x_train, x_test, y_train, y_test = train_test_split(datos, etiquetas, test_size=0.2, random_state=42)
        joints_1_train, joints_1_test = train_test_split(joint_coords_1, test_size=0.2, random_state=42)
        joints_2_train, joints_2_test = train_test_split(joint_coords_2, test_size=0.2, random_state=42)
        
        # Reajustar las formas de los datos
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        joints_1_train = np.expand_dims(joints_1_train, axis=-1)
        joints_1_test = np.expand_dims(joints_1_test, axis=-1)
        joints_2_train = np.expand_dims(joints_2_train, axis=-1)
        joints_2_test = np.expand_dims(joints_2_test, axis=-1)

        # Convertir etiquetas a numéricas si son cadenas
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_test = label_encoder.transform(y_test)

        # Extract timesteps and num_features from training data
        timesteps = x_train.shape[1]
        num_features = joints_1_train.shape[1]
        
        # Build and train the model
        modelo = construir_LSTM(timesteps, num_features)
        if modelo:
            print('modelo construido')
            modelo_entrenado = entrenar_LSTM(modelo, x_train, joints_1_train, joints_2_train, y_train, x_test, joints_1_test, joints_2_test, y_test, folder_path)
            print('modelo entrenado finalizado')
            if modelo_entrenado is not None:
                    print("empezar el umbral")
                    umbrales = obtener_umbral_por_etiqueta(modelo_entrenado, x_train, joints_1_train, joints_2_train, y_train)
                    print("Umbrales por etiqueta:", umbrales)               
                    #guardar_umbrales_en_carpeta(umbrales, folder_path, "umbrales", model_number)
        
        end_time = time.time()
        duration = end_time - start_time
        
        error_occurred = True
        end_time = time.time()
        duration = end_time - start_time if end_time else None

        # Guardar información de tiempo siempre
        if end_time and duration is not None:
            print(f"Duración total: {duration:.2f} segundos")
            with open("duracion_LSTM.txt", "a") as file:
                file.write(f"Inicio: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
                file.write(f"Fin: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n")
                file.write(f"Duración total: {duration:.6f} segundos\n")
                file.write(f"Estado: {'Éxito' if not error_occurred else 'Error'}\n")
                file.write("=" * 50 + "\n")