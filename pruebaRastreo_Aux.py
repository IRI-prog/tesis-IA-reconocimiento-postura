
import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance
from collections import deque
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class PersonTracker:
    def __init__(self):
        self.selected_id = None
        self.last_known_position = None
        self.last_keypoints = None
        self.reference_box = None
        self.reference_keypoints = None
        self.frames_missing = 0
        self.max_frames_missing = 30  # Máximo de frames perdidos antes de resetear
        self.MIN_REID_DISTANCE = 50  # Distancia máxima en píxeles para re-identificar
        self.POSITION_THRESHOLD = 0.2
        self.KEYPOINT_SIMILARITY_THRESH = 0.7
        
def FuncionP(video, base_directory):
    print("empezando el procesamiento del video")
        
    def save_keypoints_to_file(file_path, keypoints, selected_person_id=None):
        print(f"Guardando keypoints en: {file_path}")
        with open(file_path, 'w') as file:
            file.write(f"Person {selected_person_id if selected_person_id else ''}:\n")
            if len(keypoints) == 0:
                file.write("0.0,0.0\n" * 17)
            else:
                for x, y, conf in keypoints:
                    if conf > 0.5:
                        file.write(f"{x:.4f},{y:.4f}\n")
                    else:
                        file.write("0.0,0.0\n")
            file.write("\n")
        print("Keypoints guardados.")

    def extract_keypoints(pose_results, box, offset ):
        #Versión mejorada para capturar más keypoints
        try:
            kpts = pose_results.keypoints.xy[0].cpu().numpy()
            confs = pose_results.keypoints.conf[0].cpu().numpy()            
            # Priorizar puntos visibles desde atrás (hombros, caderas, rodillas, tobillos)
            back_kpt_indices = [5, 6, 11, 12, 13, 14, 15, 16]  # Hombros, caderas, piernas            
            if kpts.shape[0] < 17:
                # Rellenar con ceros solo los puntos frontales
                new_kpts = np.zeros((17, 2))
                new_confs = np.zeros(17)
                for i in back_kpt_indices:
                    if i < kpts.shape[0]:
                        new_kpts[i] = kpts[i]
                        new_confs[i] = confs[i]
                kpts = new_kpts
                confs = new_confs                
        except Exception as e:
            print(f"Error en keypoints: {str(e)}")
            return np.zeros((17, 3))
        
        adjusted = np.zeros((17, 3))
        for i in range(17):
            adjusted[i] = [
                kpts[i][0] + offset[0],
                kpts[i][1] + offset[1],
                confs[i] if (i < len(confs)) else 0.0
            ]
        return adjusted
        
    def draw_pose(frame, keypoints):
        # Conexiones de los puntos para formar el esqueleto
        skeleton = [
            (5, 7), (7, 9),    # Brazo derecho
            (6, 8), (8, 10),   # Brazo izquierdo
            (11, 13), (13, 15),    # Pierna derecha
            (12, 14), (14, 16)     # Pierna izquierda
        ]
        
        # Dibujar los puntos clave
        for kp in keypoints:
            x, y, conf = kp
            if conf > 0.5:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        for pair in skeleton:
            partA, partB = pair
            if keypoints.shape[0] > max(partA, partB) and keypoints[partA][2] > 0.5 and keypoints[partB][2] > 0.5:
                xA, yA = keypoints[partA][0], keypoints[partA][1]
                xB, yB = keypoints[partB][0], keypoints[partB][1]
                cv2.line(frame, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 255), 2)        
        return frame
     
    def process_video(video_path, yolo_tracker, yolo_pose, output_base_path):
        try:
            if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                print(f"Archivo de video no válido: {video_path}")
                return            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error al abrir el video: {video_path}")
                return
            
            # Configuración inicial
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_folder = os.path.join(output_base_path, video_name)
            os.makedirs(output_folder, exist_ok=True)
            images_folder = os.path.join(output_folder, "imagenes")
            txt_folder = os.path.join(output_folder, "txt")
            os.makedirs(images_folder, exist_ok=True)
            os.makedirs(txt_folder, exist_ok=True)
            
            # Parámetros configurables
            KEYPOINT_CONF_THRESH = 0.5
            MAX_FRAMES_SIN_DETECCION = 50
            TARGET_FPS = 10
            REID_CONFIRMATION_FRAMES = 3  # Frames necesarios para confirmar re-identificación

            # Estado del tracking
            tracking_state = {
                'selected_id': None,          # ID de la persona a seguir
                'reference_kpts': None,       # Keypoints de referencia
                'frames_sin_deteccion': 0,   # Contador de frames sin detección
                'reid_candidate': None,       # Candidato para re-identificación
                'reid_confidence': 0,         # Confianza de re-identificación
                'geometric_signature': None   # Firma geométrica de la persona
            }

            # Configuración de FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps // TARGET_FPS))
            frame_count = 0

            # Función para cálculo de firma geométrica
            def calculate_geometric_signature(kpts):
                valid = kpts[:, 2] > KEYPOINT_CONF_THRESH
                signature = []                
                # Anchura de hombros
                shoulder_width = 0.0
                if valid[5] and valid[6]:
                    shoulder_width = np.linalg.norm(kpts[5,:2] - kpts[6,:2])
                signature.append(shoulder_width)                
                # Relación hombro-cadera
                hip_ratio = 0.0
                if valid[5] and valid[6] and valid[11] and valid[12]:
                    hip_width = np.linalg.norm(kpts[11,:2] - kpts[12,:2])
                    hip_ratio = shoulder_width / hip_width if hip_width != 0 else 0
                signature.append(hip_ratio)                
                # Ángulo postural
                angle = 0.0
                if valid[5] and valid[11] and valid[13]:
                    vec1 = kpts[11] - kpts[5]
                    vec2 = kpts[13] - kpts[11]
                    angle = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
                    angle = np.degrees(angle)
                signature.append(angle)
                return np.array(signature)

            # Función de similitud geométrica
            def geometric_similarity(ref_sig, cand_sig):
                min_len = min(len(ref_sig), len(cand_sig))
                ref = ref_sig[:min_len]
                cand = cand_sig[:min_len]
                return np.linalg.norm(ref - cand)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                deteccion_valida = False
                if frame_count % frame_interval == 0:
                    # Detección de personas
                    tracking_results = yolo_tracker.track(
                        frame,
                        persist=True,
                        classes=0,
                        conf=0.5,
                        verbose=False,
                        tracker="botsort.yaml"
                    )
                    
                    # Obtener cajas e IDs
                    boxes = tracking_results[0].boxes.xyxy.cpu().numpy() if tracking_results[0].boxes.id is not None else []
                    track_ids = tracking_results[0].boxes.id.cpu().numpy().astype(int) if tracking_results[0].boxes.id is not None else []
                    
                    # Seleccionar la primera persona si aún no se ha identificado
                    if tracking_state['selected_id'] is None:
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = map(int, box)
                            roi = frame[y1:y2, x1:x2]
                            
                            if roi.size == 0:
                                continue
                                
                            # Detectar keypoints
                            pose_results = yolo_pose(roi, conf=0.3, verbose=False)[0]
                            kpts = extract_keypoints(pose_results, box, (x1, y1))
                            
                            # Verificar keypoints válidos
                            if np.sum(kpts[:,2] > KEYPOINT_CONF_THRESH) >= 3:
                                tracking_state['selected_id'] = track_ids[i]
                                tracking_state['reference_kpts'] = kpts.copy()
                                tracking_state['geometric_signature'] = calculate_geometric_signature(kpts)
                                print(f"Persona inicial seleccionada - ID: {track_ids[i]}")
                                deteccion_valida = True
                                break
                    
                    # Seguimiento de la persona seleccionada
                    if tracking_state['selected_id'] is not None:
                        # Caso 1: La persona está presente en este frame
                        if tracking_state['selected_id'] in track_ids:
                            idx = np.where(track_ids == tracking_state['selected_id'])[0][0]
                            box = boxes[idx]
                            x1, y1, x2, y2 = map(int, box)
                            roi = frame[y1:y2, x1:x2]
                            
                            if roi.size > 0:
                                pose_results = yolo_pose(roi, conf=0.3, verbose=False)[0]
                                new_kpts = extract_keypoints(pose_results, box, (x1, y1))
                                
                                if np.sum(new_kpts[:,2] > KEYPOINT_CONF_THRESH) >= 3:
                                    tracking_state['reference_kpts'] = new_kpts.copy()
                                    tracking_state['geometric_signature'] = calculate_geometric_signature(new_kpts)
                                    tracking_state['frames_sin_deteccion'] = 0
                                    tracking_state['reid_candidate'] = None
                                    tracking_state['reid_confidence'] = 0
                                    deteccion_valida = True
                        
                        # Caso 2: La persona no está presente - Buscar re-identificación
                        else:
                            tracking_state['frames_sin_deteccion'] += 1
                            
                            # Solo buscar re-identificación si ha pasado un tiempo razonable
                            if tracking_state['frames_sin_deteccion'] > 10:
                                best_candidate = None
                                min_distance = float('inf')
                                
                                for i, box in enumerate(boxes):
                                    x1, y1, x2, y2 = map(int, box)
                                    roi = frame[y1:y2, x1:x2]
                                    
                                    if roi.size == 0:
                                        continue
                                        
                                    # Detectar keypoints del candidato
                                    pose_results = yolo_pose(roi, conf=0.3, verbose=False)[0]
                                    candidate_kpts = extract_keypoints(pose_results, box, (x1, y1))
                                    
                                    if np.sum(candidate_kpts[:,2] > KEYPOINT_CONF_THRESH) >= 3:
                                        candidate_sig = calculate_geometric_signature(candidate_kpts)
                                        distance = geometric_similarity(
                                            tracking_state['geometric_signature'],
                                            candidate_sig
                                        )
                                        
                                        # Actualizar mejor candidato
                                        if distance < min_distance:
                                            min_distance = distance
                                            best_candidate = (i, candidate_kpts, candidate_sig)
                                
                                # Procesar candidato encontrado
                                if best_candidate is not None and min_distance < 0.3:
                                    i, kpts, sig = best_candidate
                                    
                                    # Si es el mismo candidato que antes, aumentar confianza
                                    if (tracking_state['reid_candidate'] is not None and 
                                        tracking_state['reid_candidate'][0] == track_ids[i]):
                                        tracking_state['reid_confidence'] += 1
                                    else:
                                        tracking_state['reid_candidate'] = (track_ids[i], kpts, sig)
                                        tracking_state['reid_confidence'] = 1
                                    
                                    # Si alcanza la confianza necesaria, aceptar re-identificación
                                    if tracking_state['reid_confidence'] >= REID_CONFIRMATION_FRAMES:
                                        print(f"Re-identificación exitosa - Nuevo ID: {track_ids[i]}")
                                        tracking_state['selected_id'] = track_ids[i]
                                        tracking_state['reference_kpts'] = kpts.copy()
                                        tracking_state['geometric_signature'] = sig
                                        tracking_state['frames_sin_deteccion'] = 0
                                        tracking_state['reid_confidence'] = 0
                                        deteccion_valida = True
                                
                                # Resetear si no hay candidatos válidos
                                else:
                                    tracking_state['reid_candidate'] = None
                                    tracking_state['reid_confidence'] = 0
                
                # Dibujar y guardar resultados solo si tenemos una persona identificada
                if deteccion_valida:
                    # Preparar nombre de archivo
                    file_name = f"frame_{frame_count:06d}"
                    txt_path = os.path.join(txt_folder, f"{file_name}.txt")
                    img_path = os.path.join(images_folder, f"{file_name}.png")                    
                    # Dibujar keypoints
                    frame = draw_pose(frame, tracking_state['reference_kpts'])
                    
                    # Guardar resultados
                    save_keypoints_to_file(txt_path, tracking_state['reference_kpts'], tracking_state['selected_id'])
                    cv2.imwrite(img_path, frame)
                elif tracking_state['reference_kpts'] is not None:
                    save_keypoints_to_file(
                    txt_path, tracking_state['reference_kpts'], tracking_state.get('selected_id', -1) )
                
                    cv2.imwrite(img_path, frame)
                #cv2.imwrite(img_path, frame)
                #save_keypoints_to_file(txt_path, np.zeros((17, 3)), tracking_state.get('selected_id'))
                frame_count += 1
            
            cap.release()
            print(f"Proceso completado. Resultados en: {output_folder}")
            return txt_folder
        
        except Exception as e:
            print(f"Error en procesamiento: {str(e)}")
            return None 
    
    yolo_tracker = YOLO("yolov8n.pt")  # Modelo ligero para tracking
    yolo_pose = YOLO("yolo11x-pose.pt")  # Modelo específico para pose
    process_video(video, yolo_tracker, yolo_pose, base_directory)
    #Procesar video y obtener carpeta de resultados
    archivo_prueba = process_video(video, yolo_tracker, yolo_pose, base_directory)
    
    import CantidadMov_Calculo as cant_prueba
    cant_prueba.funcionPrincipalCalculoArchivo(archivo_prueba, base_directory)
    
    
if __name__ == "__main__":

    start_time = time.time()  # Iniciar temporizador

    
    #FuncionP(video, base_directory)
    FuncionP()
    end_time = time.time()  # Fin del temporizador
    duration = end_time - start_time  # Cálculo del tiempo total

    # Guardar información en un archivo de texto
    with open("duracion_tiempo_ prueba de rastreo.txt", "a") as file:
        file.write(f"Inicio: {start_time:.6f} s\n")
        file.write(f"Fin: {end_time:.6f} s\n")
        file.write(f"Duración total: {duration:.6f} s\n")
        file.write("=" * 30 + "\n")
    print(f"Tiempo total de ejecución: {duration:.6f} segundos")