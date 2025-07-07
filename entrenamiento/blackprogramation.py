"""


import os
import re
import math
import json
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
# En la sección de imports:
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError  
from tensorflow.keras.metrics import Precision, Recall, Accuracy
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silencia todos los logs de TensorFlow

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, joints_2_test=None, y_test=None):
    #def __init__(self, x_test, joints_1_test, joints_2_test, y_test):
        super().__init__()
        self.x_test = x_test
        self.joints_1_test = joints_1_test
        self.joints_2_test = joints_2_test
        self.y_test = y_test
        
        # Inicializar arrays individuales (opcional)
        self.f1_values = []
        self.recall_values = []
        self.precision_values = []
        self.accuracy_values = []
        self.mse_values = []
        self.mae_values = []
        self.r2_values = []
        self.correlation_values = []
        self.val_f1_values = []
        self.val_recall_values = []
        self.val_precision_values = []
        self.val_accuracy_values = []
        self.val_mse_values = []
        self.val_mae_values = []
        self.val_r2_values = []
        self.val_correlation_values = []
        
        # Crear un diccionario para almacenar todas las métricas
        self.metrics_history = {
            "f1_score": [],
            "recall": [],
            "precision": [],
            "accuracy": [],
            "mse": [],
            "mae": [],
            "r2": [],
            "correlation": [],
            "val_f1_score": [],
            "val_recall": [],
            "val_precision": [],
            "val_accuracy": [],
            "val_mse": [],
            "val_mae": [],
            "val_r2": [],
            "val_correlation": []
        }

    def on_epoch_end(self, epoch, logs=None):
        # Obtener las predicciones del modelo
        y_pred = self.model.predict([self.x_test, self.joints_1_test, self.joints_2_test])
        # Normalizar (suponiendo que se necesita para métricas de regresión)
        y_test_normalized = (self.y_test - np.min(self.y_test)) / (np.max(self.y_test) - np.min(self.y_test))
        y_pred_normalized = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
        
        # Convertir predicciones a clases
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = self.y_test  # Asumiendo que son enteros
        
        # Calcular métricas (asegúrate de importar las funciones necesarias, por ejemplo, de sklearn.metrics)
        from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
        
        f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
        precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
        accuracy = np.mean(y_pred_classes == y_true_classes)
        mse = mean_squared_error(y_test_normalized, y_pred_normalized)
        mae = mean_absolute_error(y_test_normalized, y_pred_normalized)
        r2 = r2_score(y_test_normalized, y_pred_normalized)
        correlation = np.corrcoef(y_test_normalized.flatten(), y_pred_normalized.flatten())[0, 1]
        
        # Almacenar métricas individuales (opcional)
        self.f1_values.append(f1)
        self.recall_values.append(recall)
        self.precision_values.append(precision)
        self.accuracy_values.append(accuracy)
        self.mse_values.append(mse)
        self.mae_values.append(mae)
        self.r2_values.append(r2)
        self.correlation_values.append(correlation)
        self.val_f1_values.append(f1)
        self.val_recall_values.append(recall)
        self.val_precision_values.append(precision)
        self.val_accuracy_values.append(accuracy)
        self.val_mse_values.append(mse)
        self.val_mae_values.append(mae)
        self.val_r2_values.append(r2)
        self.val_correlation_values.append(correlation)
        
        # Actualizar el diccionario metrics_history
        self.metrics_history["f1_score"].append(f1)
        self.metrics_history["recall"].append(recall)
        self.metrics_history["precision"].append(precision)
        self.metrics_history["accuracy"].append(accuracy)
        self.metrics_history["mse"].append(mse)
        self.metrics_history["mae"].append(mae)
        self.metrics_history["r2"].append(r2)
        self.metrics_history["correlation"].append(correlation)
        self.metrics_history["val_f1_score"].append(f1)
        self.metrics_history["val_recall"].append(recall)
        self.metrics_history["val_precision"].append(precision)
        self.metrics_history["val_accuracy"].append(accuracy)
        self.metrics_history["val_mse"].append(mse)
        self.metrics_history["val_mae"].append(mae)
        self.metrics_history["val_r2"].append(r2)
        self.metrics_history["val_correlation"].append(correlation)
        
        # Imprimir las métricas para la época
        print(f"\nEpoch {epoch+1}: F1 = {f1:.4f}, Recall = {recall:.4f}, Precision = {precision:.4f}, Accuracy = {accuracy:.4f}")
        print(f"MSE = {mse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}, Corr = {correlation:.4f}")
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        print("Confusion Matrix:\n", cm)


def plot_metrics(metrics_callback, history, image_path):
    metrics = metrics_callback.metrics_history
    # Número de métricas y disposición en columnas
    num_metrics = len(metrics)
    cols = 3  # Número de columnas deseadas
    rows = math.ceil(num_metrics / cols)  # Calcular cuántas filas son necesarias
    # Crear lienzo para métricas generales
    figure = plt.figure(figsize=(5 * cols, 5 * rows))  # Ajustar el tamaño según el número de columnas y filas
    
    for i, (metric, values) in enumerate(metrics.items(), 1):
        plt.subplot(rows, cols, i)  # Subplot dinámico según las filas y columnas
        plt.plot(values, label=f"{metric} (Entrenamiento)", color='blue')
        # Validación (si existe en el historial de métricas de validación)
        val_metric = f"val_{metric}"
        if val_metric in metrics_callback.metrics_history:
            print('entra a val_metric', val_metric)
            plt.plot(metrics_callback.metrics_history[val_metric], label=f"{metric} (Validación)", color='orange')
        plt.xlabel("Épocas")
        plt.ylabel("Valores")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    # Guardar la imagen generada (antes de plt.show())
    image_path_otra = save_metrics_image(image_path, "Metricas_imagenes", figure)
    print(f"Imagen de métricas guardada en {image_path_otra}")
    return image_path_otra
    
def get_next_filename(output_base_folder, base_name):
    print('empieza a obtener el nombre del archivo')
    output_base_folder = os.path.join(folder_path, 'modelo_backpropagation')
    os.makedirs(output_base_folder, exist_ok=True)
    existing_files = [f for f in os.listdir(output_base_folder) if f.startswith(base_name)]
    max_num = 0
    for file in existing_files:
        try:
            num = int(file[len(base_name) + 1:].split('_')[0])
            max_num = max(max_num, num)
        except ValueError:
            continue
    next_num = max_num + 1
    print('termina de obtener el nombre del archivo',os.path.join(output_base_folder, f"{base_name} {next_num:02d}_modelo_backpropagation.keras") )
    return os.path.join(output_base_folder, f"{base_name} {next_num:02d}_modelo_backpropagation.keras")

def save_metrics_image(folder_path, base_name, figure):
    output_base_folder = os.path.join(folder_path, 'Metricas_imagenes') 
    os.makedirs(output_base_folder, exist_ok=True)  # Evita error si la carpeta ya existe    
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
    image_path = os.path.join(output_base_folder, f"{base_name} {next_num:02d}_modelo_backpropagation.png")
    # Verificar si la figura es válida antes de guardarla
    if isinstance(figure, plt.Figure):
        figure.savefig(image_path)
        plt.close(figure)
        return output_base_folder
    else:
        raise ValueError("El objeto figure no es una instancia válida de matplotlib.figure.Figure")
    
def buscar_archivos_csv(carpeta):
    archivos_csv = []
    for root, dirs, files in os.walk(carpeta):
        for file in files:
            if file.endswith('.csv'):
                archivos_csv.append(os.path.join(root, file))
    return archivos_csv

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

def cargar_datos_por_categoria(ruta_archivos):
    try:
        # Asegúrate de inicializar como listas vacías
        datos_completos = []  # Lista vacía en lugar de None
        etiquetas = []  # Lista vacía en lugar de None
        joint_coords_1 = []  # Lista vacía en lugar de None
        joint_coords_2 = []  # Lista vacía en lugar de None
        max_length = 0

        for archivo, etiqueta in ruta_archivos.items():
            if os.path.exists(archivo):
                datos = pd.read_csv(archivo, encoding='ISO-8859-1')

                # Reemplazar valores NaN por 0 en las columnas seleccionadas
                columnas_interes = ["Cantidad Movimiento", "Promedio Actual", "Promedio Siguiente", "Desviacion Actual", 
                                    "Desviacion Siguiente", "Homogeneidad Actual", "Homogeneidad Siguiente", "Simetria S_J"]
                datos[columnas_interes] = datos[columnas_interes].fillna(0)

                # Extraer datos seleccionados
                datos_seleccionados = datos[columnas_interes].values
                joint_paths_1 = datos['Archivo Actual'].values
                joint_paths_2 = datos['Archivo Siguiente'].values

                # Extraer las coordenadas de los joints
                coords_1 = [extract_joints_from_txt(path) for path in joint_paths_1]
                coords_2 = [extract_joints_from_txt(path) for path in joint_paths_2]

                # Encontrar la longitud máxima de las coordenadas
                max_length = max(max_length, max(len(c) for c in coords_1), max(len(c) for c in coords_2))

                # Agregar los datos a las listas correspondientes
                datos_completos.append(datos_seleccionados)
                etiquetas.extend([etiqueta] * len(datos_seleccionados))
                joint_coords_1.extend(coords_1)
                joint_coords_2.extend(coords_2)

        # Rellenar las coordenadas de los joints para que todas tengan la misma longitud
        joint_coords_1 = [np.pad(c, (0, max_length - len(c)), 'constant') for c in joint_coords_1]
        joint_coords_2 = [np.pad(c, (0, max_length - len(c)), 'constant') for c in joint_coords_2]

        datos_completos = np.vstack(datos_completos)  # (muestras, 8)
        joint_coords_1 = np.vstack(joint_coords_1)    # (muestras, 68)
        joint_coords_2 = np.vstack(joint_coords_2)    # (muestras, 68)        
        return datos_completos, np.array(etiquetas), joint_coords_1, joint_coords_2
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None, None, None, None

def normalizar_joints(joints):
    if joints.shape[0] > 0:
        scaler = MinMaxScaler()
        return scaler.fit_transform(joints)
    return joints

def construir_modelo(input_shape, joint_shape):
    try:
        # Capas de entrada
        input_data = Input(shape=(8,), name='input_data')      # (None, 8)
        input_joints_1 = Input(shape=(68,), name='joints_1')   # (None, 68)
        input_joints_2 = Input(shape=(68,), name='joints_2')   # (None, 68)
        
        # Capas densas
        x = Dense(128, activation='relu')(input_data)
        x_j1 = Dense(128, activation='relu')(input_joints_1)
        x_j2 = Dense(128, activation='relu')(input_joints_2)
        
        # Concatenar y capas finales
        merged = tf.keras.layers.concatenate([x, x_j1, x_j2])
        merged = Dense(64, activation='relu')(merged)
        output = Dense(4, activation='softmax')(merged)  # 4 clases
        
        # Compilar modelo
        modelo = Model(inputs=[input_data, input_joints_1, input_joints_2], outputs=output)
        modelo.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')])
        return modelo
    except Exception as e:
        print(f"Error al construir el modelo: {e}")
        return None

def entrenar_modelo(modelo, x_train, joints_1_train, joints_2_train, y_train, x_test, joints_1_test, joints_2_test, y_test, folder_path):
    try:
        print('empieza a entrenar el modelo',folder_path )
        # Verificar y limpiar los datos antes de entrenar
        x_train = np.asarray(x_train).astype(np.float32)
        joints_1_train = np.asarray(joints_1_train).astype(np.float32)
        joints_2_train = np.asarray(joints_2_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.int32)
        x_test = np.asarray(x_test).astype(np.float32)
        joints_1_test = np.asarray(joints_1_test).astype(np.float32)
        joints_2_test = np.asarray(joints_2_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.int32)
        
        perdida, exactitud = modelo.evaluate([x_test, joints_1_test, joints_2_test], y_test)
        print(f"Pérdida: {perdida}, Exactitud: {exactitud}")
        metrics_callback = MetricsCallback([x_test, joints_1_test, joints_2_test], y_test)

        history = modelo.fit(
            [x_train, joints_1_train, joints_2_train],  # Debe haber solo 3 elementos en esta lista
            y_train,
            epochs=90,
            batch_size=32,
            validation_data=([x_test, joints_1_test, joints_2_test], y_test)
        )

        # Guardar el modelo entrenado con un nombre único
        modelo_path = get_next_filename(folder_path, "modelo_backpropagation")
        modelo.save(modelo_path)
        print(f"Modelo guardado en {modelo_path}")
        
        #graficar_metricas(metrics_callback, history)
        plot_metrics(metrics_callback, history, folder_path)
        return modelo
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")
        return None

# Guardar los umbrales en un archivo JSON
def guardar_umbrales_en_carpeta(umbrales, carpeta_modelo, base_name):
    print('empieza a guardar los umbrales')
    try:
        output_base_folder = os.path.join(carpeta_modelo, 'modelo_backpropagation')
        os.makedirs(output_base_folder, exist_ok=True)        
        existing_files = [f for f in os.listdir(output_base_folder) if f.startswith(base_name)]
        max_num = 0
        for f in existing_files:
            match = re.findall(r"(\d+)_Back_umbrales\.json$", f)
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
    predicciones = modelo.predict([x_test, joints_1_test, joints_2_test])
    umbrales = {}
    for i, etiqueta in enumerate(np.unique(y_test)):
        umbrales[etiqueta] = (np.percentile(predicciones[:, i], 5), np.percentile(predicciones[:, i], 95))
    return umbrales


if __name__ == "__main__":
    carpeta =  "seguimiento/datos agrupar/cantidadMovimiento"
    archivos_csv = buscar_archivos_csv(carpeta)
    if not archivos_csv:
        print("No se encontraron archivos .csv en la carpeta especificada.")
        exit()    
    ruta_archivos = {}
    for archivo in archivos_csv:
        print("Archivo a analizar: ", archivo)
        if "Caminando" in archivo:
            ruta_archivos[archivo] = "Caminando"
        elif "Parado" in archivo:
            ruta_archivos[archivo] = "Parado"
        elif "Sentado" in archivo:
            ruta_archivos[archivo] = "Sentado"
        elif "Sin personas" in archivo:
            ruta_archivos[archivo] = "Sin personas"
    
    datos_nuevos, etiquetasN, joint_paths_1N, joint_paths_2N = cargar_datos_por_categoria(ruta_archivos)
    
    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    etiquetasN = label_encoder.fit_transform(etiquetasN)

    folder_path = "seguimiento/datos agrupar/modelos_entrenados"
    if datos_nuevos is not None:
        x_train, x_test, joints_1_train, joints_1_test, joints_2_train, joints_2_test, y_train, y_test = train_test_split(
            datos_nuevos, joint_paths_1N, joint_paths_2N, etiquetasN, test_size=0.2, random_state=42)
        #x_train, x_test, y_train, y_test = train_test_split(datos_nuevos, etiquetasN, test_size=0.2, random_state=42)
        #joints_1_train, joints_1_test = train_test_split(joint_paths_1N, test_size=0.2, random_state=42)
        #joints_2_train, joints_2_test = train_test_split(joint_paths_2N, test_size=0.2, random_state=42)
        
        modelo = construir_modelo(x_train.shape[1], joints_1_train.shape[1])
        if modelo:
            modelo_entrenado = entrenar_modelo(modelo, x_train, joints_1_train, joints_2_train, y_train, x_test, joints_1_test,  joints_2_test, y_test,folder_path)
            modelo_entrenado = entrenar_modelo( modelo, x_train, joints_1_train, joints_2_train, y_train, x_test, joints_1_test, 
            joints_2_test, y_test, folder_path )
            if modelo_entrenado:
                umbrales = obtener_umbral_por_etiqueta(modelo_entrenado, x_test, joints_1_test, joints_2_test, y_test)
                print("Umbrales por etiqueta:", umbrales)
                guardar_umbrales_en_carpeta(umbrales, folder_path, "umbrales")
                
                
"""


import os
import re
import math
import json
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import label_binarize
# En la sección de imports:
from tensorflow.keras.metrics import Accuracy, Precision, Recall, MeanSquaredError, MeanAbsoluteError

from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Calcular métricas (asegúrate de importar las funciones necesarias, por ejemplo, de sklearn.metrics)
from sklearn.metrics import classification_report, f1_score, confusion_matrix, recall_score, precision_score, f1_score, accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import  accuracy_score, confusion_matrix

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silencia todos los logs de TensorFlow
tf.config.set_visible_devices([], 'GPU')



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
        self.csv_path = os.path.join(folder_path, "metricas_por_epoca_Back.csv")
        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=self.metrics_history.keys()).to_csv(self.csv_path, index=False)        
        # Preparar carpetas para gráficas
        self.preparar_directorios()
        

    def preparar_directorios(self):
        """Crea carpetas para guardar gráficas y matrices de confusión."""
        self.reg_plots_dir = os.path.join(self.folder_path, "regression_metrics_plots_Back")
        self.cm_plots_dir = os.path.join(self.folder_path, "confusion_matrices_Back")
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
        np.savetxt(os.path.join(self.folder_path, "confusion_matrix_final_Back.csv"), 
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
        plt.title("Matriz de Confusión - Conjunto de Prueba Final en Backpropagetion", fontsize=14)
        plt.xlabel("Predicciones", fontsize=12)
        plt.ylabel("Valores Reales", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        # Guardar en alta resolución
        plt.savefig(os.path.join(self.folder_path, "confusion_matrix_final_Back.png"), 
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


def graficar_evolucion_matrices(metrics_callback, folder_path):
    plt.figure(figsize=(12, 8))    
    output_base_folder = os.path.join(folder_path, 'Evolucion_cm_epoch_version_Back')
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
    # Configurar figura única
    fig = plt.figure(figsize=(25, 30))
    # Versión corregida (6 filas = 6 valores en height_ratios)
    gs = fig.add_gridspec(6, 2, height_ratios=[1.5, 1.5, 1.5, 1, 1, 1])
    
    # Subplots principales
    ax0 = fig.add_subplot(gs[0, 0])  # F1 por clase
    ax1 = fig.add_subplot(gs[0, 1])  # Macro F1 
    ax2 = fig.add_subplot(gs[1, 0])  # Accuracy
    ax3 = fig.add_subplot(gs[1, 1])  # Precisión por clase
    ax4 = fig.add_subplot(gs[2, 0])  # Recall por clase
    ax5 = fig.add_subplot(gs[2, 1])  # Macro Recall 
    
    # Mini subplots para métricas macro
    ax6 = fig.add_subplot(gs[3, 0])  # Loss
    ax7 = fig.add_subplot(gs[3, 1])  # MSE
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

    # 7. Nuevo: Macro F1
    ax1.plot(epochs, metrics_callback.metrics_history["f1_score"], 'green', linewidth=2)
    ax1.set_title('F1-Score Global', fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.2)

     # 4. Precisión por clase (Original)
    for i in range(metrics_callback.num_classes):
        ax2.plot(epochs, metrics_callback.metrics_history[f"precision_{i}"], 
                color=colors[i], linestyle='--', alpha=0.7)
    ax2.set_title('Precisión por Clase', fontsize=12)
    ax2.grid(alpha=0.2)

     # 2. Accuracy (Original)
    ax3.plot(history.epoch, history.history['accuracy'], 'navy', label='Entrenamiento')
    ax3.plot(history.epoch, history.history['val_accuracy'], 'skyblue', label='Validación')
    ax3.set_title('Precisión Global', fontsize=12)
    ax3.legend()
    ax3.grid(alpha=0.2)

     # 5. Recall por clase (Original)
    for i in range(metrics_callback.num_classes):
        ax4.plot(epochs, metrics_callback.metrics_history[f"recall_{i}"], 
                color=colors[i], linestyle='-.', alpha=0.7)
    ax4.set_title('Recall por Clase', fontsize=12)
    ax4.grid(alpha=0.2)
    
    # 8. Nuevo: Macro Recall
    ax5.plot(epochs, metrics_callback.metrics_history["recall"], 'blue', linewidth=2)
    ax5.set_title('Recall Global', fontsize=10)
    ax5.set_ylim(0, 1)
    ax5.grid(alpha=0.2)

    # 3. Loss (Original)
    ax6.plot(history.epoch, history.history['loss'], 'darkred', label='Entrenamiento')
    ax6.plot(history.epoch, history.history['val_loss'], 'lightcoral', label='Validación')
    ax6.set_title('Pérdida', fontsize=12)
    ax6.legend()
    ax6.grid(alpha=0.2) 

    # 6. Métricas de regresión (Original)
    ax7.plot(epochs, metrics_callback.metrics_history["mse"], 'purple')
    ax7.set_title('MSE', fontsize=12)
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
    plt.savefig(folder_path, dpi=300, bbox_inches='tight')
   
    # Guardar solo una vez
    os.makedirs(folder_path, exist_ok=True)
    image_path = save_metrics_image(folder_path, "metricas_Backpropagetion", fig)
    image_path_specific = os.path.join(folder_path, 'metricas_Backpropagetion.png')
    fig.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
    # Cerrar figura explícitamente
    plt.close(fig)
  
    return image_path_specific
    
def get_next_filename(output_base_folder, base_name):
    print('empieza a obtener el nombre del archivo')
    output_base_folder = os.path.join(folder_path, 'modelo_backpropagation')
    os.makedirs(output_base_folder, exist_ok=True)
    existing_files = [f for f in os.listdir(output_base_folder) if f.startswith(base_name)]
    max_num = 0
    for file in existing_files:
        try:
            num = int(file[len(base_name) + 1:].split('_')[0])
            max_num = max(max_num, num)
        except ValueError:
            continue
    next_num = max_num + 1
    print('termina de obtener el nombre del archivo',os.path.join(output_base_folder, f"{base_name} {next_num:02d}_modelo_backpropagation.keras") )
    return os.path.join(output_base_folder, f"{base_name} {next_num:02d}_modelo_backpropagation.keras")

def save_metrics_image(folder_path, base_name, figure):
    print('empieza a guardar metricas')
    output_base_folder = os.path.join(folder_path, 'Metricas_imagenes_Back') 
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
    image_name = os.path.join(output_base_folder, f"{base_name}_{next_num:02d}_modelo_backpropagation.png")
    
    if isinstance(figure, plt.Figure):
        figure.savefig(image_name)
        plt.close(figure)
        return image_name
    else:
        raise ValueError("El objeto figure no es una instancia válida de matplotlib.figure.Figure")
    
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

def _ensure_2d(data):
    return np.atleast_2d(data)

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

def normalizar_joints(joints):
    if joints.shape[0] > 0:
        scaler = MinMaxScaler()
        return scaler.fit_transform(joints)
    return joints

def construir_modelo(input_shape, joint_shape):
    try:
        # Entradas
        input_data = Input(shape=(input_shape,))
        input_j1 = Input(shape=(joint_shape,))
        input_j2 = Input(shape=(joint_shape,))
    
        # Capas con regularización
        x = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(input_data)
        x = Dropout(0.3)(x)
        
        x_j1 = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(input_j1)
        x_j1 = BatchNormalization()(x_j1)
        
        x_j2 = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(input_j2)
        x_j2 = BatchNormalization()(x_j2)
        
        merged = concatenate([x, x_j1, x_j2])
        merged = Dense(256, activation="relu")(merged)
        merged = Dropout(0.4)(merged)
        output = Dense(4, activation="softmax")(merged)
        
        # Compilar modelo
        modelo = Model(inputs=[input_data, input_j1, input_j2], outputs=output)
        modelo.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return modelo
    except Exception as e:
        print(f"Error al construir el modelo: {e}")
        return None

def entrenar_modelo(modelo, x_train, joints_1_train, joints_2_train, y_train, x_test, 
                    joints_1_test, joints_2_test, y_test, folder_path):
    try:
        print('empieza a entrenar el modelo',folder_path )  
        # Antes de llamar a modelo.fit y modelo.evaluate, ajusta las dimensiones:
        joints_1_train = np.squeeze(joints_1_train, axis=1)
        joints_2_train = np.squeeze(joints_2_train, axis=1)
        joints_1_test = np.squeeze(joints_1_test, axis=1)
        joints_2_test = np.squeeze(joints_2_test, axis=1)

        x_train = np.asarray(x_train).astype(np.float32)
        joints_1_train = np.asarray(joints_1_train).astype(np.float32)
        joints_2_train = np.asarray(joints_2_train).astype(np.float32)
     
        y_train = np.asarray(y_train).astype(np.int32).flatten()
        y_test = np.asarray(y_test).astype(np.int32).flatten()
        joints_1_test = np.asarray(joints_1_test).astype(np.float32)
        joints_2_test = np.asarray(joints_2_test).astype(np.float32)
        y_test = np.asarray(y_test).astype(np.int32)
        
        perdida, exactitud = modelo.evaluate([x_test, joints_1_test, joints_2_test], y_test)
        print(f"Pérdida: {perdida}, Exactitud: {exactitud}")
    
        # Crear el callback de métricas
        metrics_callback = MetricsCallback(x_test, joints_1_test, joints_2_test, y_test, folder_path)
        #early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)  
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Entrenar
        history = modelo.fit(
            [x_train, joints_1_train, joints_2_train],
            y_train,  epochs=100, batch_size=32,
            validation_data=([x_test, joints_1_test, joints_2_test], y_test),
            #callbacks=[metrics_callback, early_stop])
            callbacks=[metrics_callback, early_stop])

        # En entrenar_modelo, después de model.fit:
        print("Claves en history.history:", history.history.keys())
            
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

        # Guardar matriz de confusión final por separado
        y_pred_final = modelo.predict([x_test, joints_1_test, joints_2_test])
        y_pred_classes = np.argmax(y_pred_final, axis=1)
        np.savetxt(os.path.join(folder_path, "matriz_confusion_final_Backpropagetion.csv"), cm_final, delimiter=",", fmt="%d")        
        
        # Guardar todas las métricas en un CSV
        df_metrics = pd.DataFrame(metrics_callback.metrics_history)
        csv_path = os.path.join(folder_path, "metricas_por_epoca_back.csv")
        df_metrics.to_csv(csv_path, index=False)
        print(f"CSV con métricas guardado en: {csv_path}")

        # Guardar el modelo entrenado con un nombre único
        modelo_path = get_next_filename(folder_path, "modelo_backpropagation")
        modelo.save(modelo_path)
        print(f"Modelo guardado en {modelo_path}")

        print("\nResultados finales en el conjunto de prueba:")
        print(f"F1-Score: {f1:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        #graficar_metricas(metrics_callback, history)
        graficar_metricas(metrics_callback, history, folder_path)
        return modelo
    except Exception as e:
        print(f"Error al entrenar el modelo: {e}")
        return None

# Guardar los umbrales en un archivo JSON
def guardar_umbrales_en_carpeta(umbrales, carpeta_modelo, base_name):
    print('empieza a guardar los umbrales')
    try:
        output_base_folder = os.path.join(carpeta_modelo, 'modelo_backpropagation')
        os.makedirs(output_base_folder, exist_ok=True)        
        existing_files = [f for f in os.listdir(output_base_folder) if f.startswith(base_name)]
        max_num = 0
        for f in existing_files:
            match = re.findall(r"(\d+)_Back_umbrales\.json$", f)
            if match:
                max_num = max(max_num, int(match[0]))
        cont= max_num + 1
        archivo_umbral = os.path.join(output_base_folder, f"{base_name} {cont + 1:02d}_umbrales.json")
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




if __name__ == "__main__":
    carpeta =  "D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet"
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
    
    datos_nuevos, etiquetasN, joint_paths_1N, joint_paths_2N = cargar_datos_por_categoria(ruta_archivos)
    
    # Codificar las etiquetas
    label_encoder = LabelEncoder()
    etiquetasN = label_encoder.fit_transform(etiquetasN)

    folder_path = "D:/programs/redNeuronal/modificaciones/datos agrupar/modelos_entrenados"
    if datos_nuevos is not None:
        x_train, x_test, joints_1_train, joints_1_test, joints_2_train, joints_2_test, y_train, y_test = train_test_split(
            datos_nuevos, joint_paths_1N, joint_paths_2N, etiquetasN, test_size=0.2, random_state=42)
       
        modelo = construir_modelo(x_train.shape[1], joints_1_train.shape[1])
        if modelo:
            modelo_entrenado = entrenar_modelo(modelo, x_train, joints_1_train, joints_2_train, y_train, x_test, joints_1_test,  joints_2_test, y_test,folder_path)
            if modelo_entrenado:
                umbrales = obtener_umbral_por_etiqueta(modelo_entrenado, x_test, joints_1_test, joints_2_test, y_test)
                print("Umbrales por etiqueta:", umbrales)
                guardar_umbrales_en_carpeta(umbrales, folder_path, "umbrales")
                