import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Usar backend compatible con Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd


def subir_video():
    try:
        ruta_video = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if ruta_video and ruta_video.endswith(('.mp4', '.avi', '.mov')):
            print(f"El video se ha seleccionado: {ruta_video}")
            import pruebaRastreo_Aux as archivo
            archivo.FuncionP(ruta_video, base_directory)
        else:
            raise ValueError("Por favor, seleccione un archivo de video válido (.mp4, .avi, .mov).")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def pedir_carpeta():
    try:
        ruta_carpeta = filedialog.askdirectory()
        if ruta_carpeta:
            print(f"La carpeta seleccionada es: {ruta_carpeta}")
            videos_encontrados = False
            for archivo in os.listdir(ruta_carpeta):
                if archivo.endswith(('.mp4', '.avi', '.mov')):
                    videos_encontrados = True
                    ruta_video = os.path.join(ruta_carpeta, archivo)
                    print(f"Procesando video: {ruta_video}")
                    import extraerTXT_Carpeta as carpeta
                    carpeta.FuncionP(ruta_video, base_directory)
            if not videos_encontrados:
                raise ValueError("No se encontraron archivos de video (.mp4, .avi, .mov) en la carpeta seleccionada.")
        else:
            raise ValueError("No se seleccionó ninguna carpeta.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def representar_graficas():
    try:
        # Construir la ruta a la carpeta "cantidadMovimiento"
        ruta_cantidad_movimiento = os.path.join(archivoPred, 'cantidadMovimiento')
        if os.path.exists(ruta_cantidad_movimiento) and os.path.isdir(ruta_cantidad_movimiento):
            print(f"Navegando automáticamente a la carpeta: {ruta_cantidad_movimiento}")
            # Importar el módulo que genera la interfaz y pasar la carpeta "cantidadMovimiento"
            import Grafica_Carpeta as repre
            repre.funcionPrincipal(ruta_cantidad_movimiento)  # Pasar la ruta directamente
        else:
            raise ValueError(f"No se encontró la carpeta 'cantidadMovimiento' en {archivoPred}.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def representar_graficas_Pred():
    try:
        # Construir la ruta a la carpeta "cantidadMovimiento"
        ruta_cantidad_movimiento = os.path.join(base_directory, 'cantidadMovimiento')
        if os.path.exists(ruta_cantidad_movimiento) and os.path.isdir(ruta_cantidad_movimiento):
            print(f"Navegando automáticamente a la carpeta: {ruta_cantidad_movimiento}")
            # Importar el módulo que genera la interfaz y pasar la carpeta "cantidadMovimiento"
            import Grafica_Carpeta as repre
            repre.funcionPrincipal(ruta_cantidad_movimiento)  # Pasar la ruta directamente
        else:
            raise ValueError(f"No se encontró la carpeta 'cantidadMovimiento' en {base_directory}.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def representar_imagenes_MC():
    try:
        # Ruta a la carpeta de matrices de confusión
        ruta_matrices = os.path.join(base_directory, 'matricesConfusion')
        if not os.path.exists(ruta_matrices) or not os.path.isdir(ruta_matrices):
            raise ValueError(f"No se encontró la carpeta 'matricesConfusion' en {base_directory}.")
        
        # Obtener lista de imágenes de matrices de confusión
        imagenes = [f for f in os.listdir(ruta_matrices) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                   and 'confusion' in f.lower()]
        
        if not imagenes:
            raise ValueError("No se encontraron imágenes de matrices de confusión en la carpeta.")
        
        # Crear ventana para mostrar imágenes
        ventana_mc = tk.Toplevel()
        ventana_mc.title("Matrices de Confusión")
        ventana_mc.geometry("800x600")
        
        # Frame para contenedor de imágenes
        frame = tk.Frame(ventana_mc)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas con scrollbar
        canvas = tk.Canvas(frame)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mostrar cada imagen
        for img_file in imagenes:
            img_path = os.path.join(ruta_matrices, img_file)
            try:
                img = Image.open(img_path)
                img.thumbnail((700, 500))  # Redimensionar manteniendo aspecto
                img_tk = ImageTk.PhotoImage(img)
                
                # Crear etiqueta para la imagen
                lbl_img = tk.Label(scrollable_frame, image=img_tk)
                lbl_img.image = img_tk  # Mantener referencia
                lbl_img.pack(pady=10)
                
                # Etiqueta con nombre del archivo
                lbl_nombre = tk.Label(scrollable_frame, text=img_file)
                lbl_nombre.pack()
                
            except Exception as img_error:
                print(f"Error cargando imagen {img_file}: {str(img_error)}")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    except Exception as e:
        messagebox.showerror("Error", str(e))
        
def mostrar_resultados_prediccion():
    try:
        # Verificar que base_directory existe
        if not os.path.exists(base_directory) or not os.path.isdir(base_directory):
            raise ValueError(f"El directorio base no existe: {base_directory}")
        
        # Crear ventana de selección
        ventana_seleccion = tk.Toplevel()
        ventana_seleccion.title("Seleccionar Archivo de Resultados")
        ventana_seleccion.geometry("600x500")
        ventana_seleccion.resizable(True, True)
        
        # Marco principal
        main_frame = tk.Frame(ventana_seleccion)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Título
        lbl_titulo = tk.Label(
            main_frame, 
            text="Seleccione un archivo de resultados",
            font=("Arial", 14, "bold")
        )
        lbl_titulo.pack(pady=(0, 15))
        
        # Marco para los botones de archivos (con desplazamiento)
        frame_archivos = tk.Frame(main_frame)
        frame_archivos.pack(fill=tk.BOTH, expand=True)
        
        # Canvas y barra de desplazamiento
        canvas = tk.Canvas(frame_archivos)
        scrollbar = tk.Scrollbar(frame_archivos, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Obtener archivos CSV
        archivos_csv = [f for f in os.listdir(base_directory) if f.endswith('.csv')]
        
        if not archivos_csv:
            lbl_vacio = tk.Label(
                scrollable_frame, 
                text="No se encontraron archivos CSV en el directorio",
                fg="gray", 
                font=("Arial", 10)
            )
            lbl_vacio.pack(pady=20)
        else:
            # Botón para cada archivo CSV
            for archivo in archivos_csv:
                btn_archivo = tk.Button(
                    scrollable_frame,
                    text=archivo,
                    width=50,
                    height=1,
                    anchor="w",
                    command=lambda f=archivo: cargar_resultados(os.path.join(base_directory, f)),  # Llamar a cargar_resultados
                    relief="flat",
                    bg="#f0f0f0",
                    font=("Arial", 9)
                )
                btn_archivo.pack(pady=5, padx=5)
        
        # Empaquetar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Botón para abrir otro directorio
        def abrir_directorio():
            nuevo_dir = filedialog.askdirectory(initialdir=base_directory)
            if nuevo_dir:
                #nonlocal base_directory
                base_directory = nuevo_dir
                ventana_seleccion.destroy()
                mostrar_resultados_prediccion()  # Recargar la ventana con nuevo directorio
        
        btn_directorio = tk.Button(
            main_frame,
            text="Cambiar Directorio",
            command=abrir_directorio,
            bg="#2196F3",
            fg="white",
            padx=10
        )
        btn_directorio.pack(pady=(15, 5))
        
        # Botón de cancelar
        btn_cancelar = tk.Button(
            main_frame,
            text="Cancelar",
            command=ventana_seleccion.destroy,
            bg="#f44336",
            fg="white",
            padx=15
        )
        btn_cancelar.pack(pady=5)
        
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al mostrar los archivos:\n{str(e)}")
        import traceback
        traceback.print_exc()

# Función para cargar y mostrar los resultados de un archivo específico
def cargar_resultados(ruta_completa):
    try:
        # Leer el archivo CSV
        df = pd.read_csv(ruta_completa)
        archivo_reciente = os.path.basename(ruta_completa)
        
        # Crear ventana para mostrar resultados
        ventana_resultados = tk.Toplevel()
        ventana_resultados.title(f"Resultados de Predicción - {archivo_reciente}")
        ventana_resultados.geometry("1000x800")
        
        # Frame principal
        main_frame = tk.Frame(ventana_resultados)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== SECCIÓN SUPERIOR: GRÁFICOS =====
        frame_graficos = tk.LabelFrame(main_frame, text="Visualización de Resultados", font=("Arial", 10, "bold"))
        frame_graficos.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Contenedor para gráficos
        container_graficos = tk.Frame(frame_graficos)
        container_graficos.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Gráfico 1: Distribución de predicciones
        frame_distribucion = tk.Frame(container_graficos)
        frame_distribucion.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        lbl_dist = tk.Label(frame_distribucion, text="Distribución de Predicciones", font=("Arial", 9, "bold"))
        lbl_dist.pack(pady=(0, 5))
        
        fig_dist = plt.Figure(figsize=(6, 4))
        ax_dist = fig_dist.add_subplot(111)
        
        # Contar predicciones por clase
        clases = ["En Movimiento", "Parado", "Sentado", "Sin Personas"]
        
        # Obtener conteos para cada modelo
        conteos = {}
        for modelo in ['LSTM', 'Backpropagation']:
            col_name = f'Prediccion_{modelo}'
            if col_name in df.columns:
                counts = df[col_name].value_counts()
                conteos[modelo] = [counts.get(clase, 0) for clase in clases]
        
        # Crear gráfico de barras
        bar_width = 0.35
        index = range(len(clases))
        
        for i, (modelo, valores) in enumerate(conteos.items()):
            pos = [x + i * bar_width for x in index]
            color = 'blue' if modelo == 'LSTM' else 'green'
            ax_dist.bar(pos, valores, bar_width, label=modelo, alpha=0.7, color=color)
        
        ax_dist.set_xlabel('Clases')
        ax_dist.set_ylabel('Cantidad')
        ax_dist.set_title('Distribución por Modelo')
        ax_dist.set_xticks([x + bar_width/2 * (len(conteos)-1) for x in index])
        ax_dist.set_xticklabels(clases)
        ax_dist.legend()
        fig_dist.tight_layout()
        
        canvas_dist = FigureCanvasTkAgg(fig_dist, master=frame_distribucion)
        canvas_dist.draw()
        canvas_dist.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Gráfico 2: Evolución temporal (si está disponible)
        frame_temporal = tk.Frame(container_graficos)
        frame_temporal.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Verificar si tenemos una columna temporal
        columna_temporal = None
        for col in ['Frame', 'Tiempo', 'Timestamp', 'Time', 'Index']:
            if col in df.columns:
                columna_temporal = col
                break
        
        if columna_temporal:
            lbl_temp = tk.Label(frame_temporal, text="Evolución Temporal", font=("Arial", 9, "bold"))
            lbl_temp.pack(pady=(0, 5))
            
            fig_temp = plt.Figure(figsize=(6, 4))
            ax_temp = fig_temp.add_subplot(111)
            
            # Ordenar por la columna temporal
            df_temp = df.sort_values(by=columna_temporal)
            
            # Graficar predicciones a lo largo del tiempo
            clase_a_numero = {clase: i for i, clase in enumerate(clases)}
            
            for modelo in ['LSTM', 'Backpropagation']:
                col_name = f'Prediccion_{modelo}'
                if col_name in df_temp.columns:
                    num_col = f'{modelo}_num'
                    df_temp[num_col] = df_temp[col_name].map(clase_a_numero)
                    color = 'blue' if modelo == 'LSTM' else 'green'
                    ax_temp.plot(df_temp[columna_temporal], df_temp[num_col], 'o-', label=modelo, 
                                alpha=0.7, markersize=4, color=color)
            
            ax_temp.set_xlabel(columna_temporal)
            ax_temp.set_ylabel('Clase')
            ax_temp.set_title('Evolución en el Tiempo')
            ax_temp.set_yticks(range(len(clases)))
            ax_temp.set_yticklabels(clases)
            ax_temp.legend()
            fig_temp.tight_layout()
            
            canvas_temp = FigureCanvasTkAgg(fig_temp, master=frame_temporal)
            canvas_temp.draw()
            canvas_temp.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        #else:
            #lbl_no_temp = tk.Label(frame_temporal, text="Sin datos temporales", font=("Arial", 9), fg="gray")
            #lbl_no_temp.pack(expand=True, pady=20)
        
        
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error al cargar el archivo:\n{str(e)}")
        import traceback
        traceback.print_exc()
    
    
# Obtener la ruta del directorio del archivo actual
current_directory = os.path.dirname(os.path.abspath(__file__))
# Definir la carpeta inicial dentro del directorio del archivo
base_directory = os.path.join(current_directory, 'datos') 
archivoPred="D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet"

# Crear la carpeta inicial si no existe
if not os.path.exists(base_directory):
    os.makedirs(base_directory)

# Creación de la ventana principal
root = tk.Tk()
root.geometry("400x200")
root.resizable(width=False, height=False)
root.title("Procesamiento de Video")

# Botones para las diferentes funcionalidades
btn_subir_video = tk.Button(root, text="Subir Video para una predicción", command=subir_video)
btn_subir_video.pack(pady=5)

#btn_seleccionar_carpeta = tk.Button(root, text="Seleccionar Carpeta", command=pedir_carpeta)
#btn_seleccionar_carpeta.pack()

btn_representar_graficas = tk.Button(root, text="Graficas de valores de entrenamiento", command=representar_graficas)
btn_representar_graficas.pack(pady=5)

btn_representar_graficas_pred = tk.Button(root, text="Graficas de valores de Prediccion", command=representar_graficas_Pred)
btn_representar_graficas_pred.pack(pady=5)

#btn_representar_imagenes_MC = tk.Button(root, text="Imagenes de matris de confusion", command=representar_imagenes_MC)
#btn_representar_imagenes_MC.pack(pady=5)

btn_mostrar_resultados = tk.Button(root, text="Mostrar resultados de predicción", command=mostrar_resultados_prediccion)
btn_mostrar_resultados.pack(pady=5)

# Ejecución de la ventana principal
root.mainloop()

