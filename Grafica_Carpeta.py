

import os
import pandas as pd
import cv2
import tkinter as tk
from tkinter import ttk, Tk, Button, messagebox
import matplotlib.pyplot as plt
import numpy as np

def funcionPrincipal(carpeta_base):
    if not os.path.isdir(carpeta_base):
        messagebox.showerror("Error", f"No se encontró la carpeta: {carpeta_base}")
        return

    def crear_interfaz_carpetas(carpeta_base):
        root = Tk()
        root.title("Visualizador de Carpetas")
        root.geometry("400x300")
        subcarpetas = [f.path for f in os.scandir(carpeta_base) if f.is_dir()]
        for sub in subcarpetas:
            Button(root, text=os.path.basename(sub),
                   command=lambda c=sub: crear_interfaz_columnas_para_csv(c)).pack(pady=5)
        Button(root, text=os.path.basename(carpeta_base),
               command=lambda c=carpeta_base: crear_interfaz_columnas_para_csv(c)).pack(pady=5)
        root.mainloop()

    def crear_interfaz_columnas_para_csv(carpeta):
        root = Tk()
        root.title(f"CSV: {os.path.basename(carpeta)}")
        root.geometry("400x300")

        cont = ttk.Frame(root); cont.pack(fill="both", expand=True)
        canvas = tk.Canvas(cont); scrollbar = ttk.Scrollbar(cont, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y"); canvas.pack(side="left", fill="both", expand=True)
        frame_int = ttk.Frame(canvas)
        canvas.create_window((0,0), window=frame_int, anchor="nw")

        archivos = [f for f in os.listdir(carpeta) if f.endswith(".csv")]
        if not archivos:
            messagebox.showinfo("Info","No hay CSV aquí."); root.destroy(); return

        for archivo in archivos:
            path = os.path.join(carpeta, archivo)
            df = pd.read_csv(path)
            # botón por cada columna que quieras graficar
            for idx in [2,7,9,10]:
                if idx < len(df.columns):
                    col = df.columns[idx]
                    Button(frame_int, text=f"{archivo} → {col}",
                           command=lambda c=col, f=path: mostrar_grafica(f,c)).pack(anchor="w", pady=2)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)),"units"))
        root.mainloop()

    def mostrar_grafica(csv_file, columna):
        df = pd.read_csv(csv_file)
        if columna not in df.columns:
            messagebox.showerror("Error", f"'{columna}' no en {csv_file}"); return

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(df[columna], picker=5)
        ax.set(xlabel="Frame index", ylabel=columna, title=columna); ax.grid(True)

        def on_pick(event):
            i = event.ind[0]
            # ahora tomo directamente las columnas "Archivo Actual" y "Archivo Siguiente"
            txt1 = str(df.at[i,"Archivo Actual"]).strip()
            txt2 = str(df.at[i,"Archivo Siguiente"]).strip()
            # construir ruta absoluta del txt
            base_dir = os.path.dirname(csv_file)
            txt_path1 = os.path.abspath(os.path.join(base_dir, txt1))
            txt_path2 = os.path.abspath(os.path.join(base_dir, txt2))
            # sustituir '/txt/' → '/imagenes/' y '.txt' → '.png'
            img1 = txt_path1.replace(os.sep+"txt"+os.sep, os.sep+"imagenes"+os.sep).replace(".txt",".png")
            img2 = txt_path2.replace(os.sep+"txt"+os.sep, os.sep+"imagenes"+os.sep).replace(".txt",".png")

            for p in (img1,img2):
                if not os.path.isfile(p):
                    messagebox.showerror("Error", f"No existe:\n{p}")
                    return
            mostrar_imagenes(img1,img2, os.path.basename(img1), os.path.basename(img2))

        fig.canvas.mpl_connect("pick_event", on_pick)
        plt.show()

    def mostrar_imagenes(p1,p2,n1,n2):
        img1 = cv2.imread(p1); img2 = cv2.imread(p2)
        if img1 is None or img2 is None:
            messagebox.showerror("Error","No se cargó imagen"); return
        img1 = cv2.resize(img1,(450,350)); img2 = cv2.resize(img2,(450,350))
        combo = cv2.hconcat([img1,img2])
        cv2.putText(combo,n1,(10,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.putText(combo,n2,(460,15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        cv2.imshow("Comparación", combo); cv2.waitKey(0); cv2.destroyAllWindows()

    crear_interfaz_carpetas(carpeta_base)

if __name__ == "__main__":
    carpeta='D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet/cantidadMovimiento'
    funcionPrincipal(carpeta)
    #funcionPrincipal()
