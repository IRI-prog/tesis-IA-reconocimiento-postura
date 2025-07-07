import os
import cv2
import numpy as np
from collections import deque
import time
from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class PersonTracker:
    def __init__(self):
        self.selected_id = None
        self.initial_features = None
        self.last_known_position = None
        self.reid_threshold = 0.7  # Umbral de similitud para re-identificación
        self.max_frames_missing = 30  # Máximo frames perdidos antes de resetear
        self.frames_missing = 0

def FuncionP(carpeta_videos, base_directory):
    
     # Validar existencia de la carpeta de videos
    if not os.path.exists(carpeta_videos):
        raise ValueError(f"La carpeta de videos no existe: {carpeta_videos}")
    # Cargar modelos una sola vez
    yolo_tracker = YOLO("yolov8n.pt")  # Modelo para tracking
    yolo_pose = YOLO("yolo11x-pose.pt")  # Modelo para pose estimation
    # Lista de extensiones de video soportadas
    extensiones_video = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    # Procesar cada archivo en la carpeta
    for archivo in os.listdir(carpeta_videos):
        video_path = os.path.join(carpeta_videos, archivo)        
        # Filtrar solo archivos de video
        if not os.path.isfile(video_path):
            continue
        # Verificar extensión del archivo
        if not archivo.lower().endswith(extensiones_video):
            print(f"\nArchivo ignorado: {archivo} (formato no soportado)")
            continue
        print(f"\n{'=' * 50}")
        print(f"Procesando video: {archivo}")
        print(f"{'=' * 50}")

        # Crear carpeta de resultados para este video
        nombre_video = os.path.splitext(archivo)[0]
        video_output_folder = os.path.join(base_directory, nombre_video)
        os.makedirs(video_output_folder, exist_ok=True)
        # Procesar el video
        try:
            process_video(video_path, yolo_tracker, yolo_pose, base_directory )
        except Exception as e:
            print(f"Error procesando {archivo}: {str(e)}")
            continue
    print("\nProcesamiento de todos los videos completado!")
   
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
                # Asegúrate de que la carpeta se crea
                os.makedirs(output_folder, exist_ok=True)  # <-- Esto ya lo haces
                images_folder = os.path.join(output_folder, "imagenes")
                txt_folder = os.path.join(output_folder, "txt")
                os.makedirs(images_folder, exist_ok=True)
                os.makedirs(txt_folder, exist_ok=True)
                print("txt_folder", txt_folder)
                archivo_prueba = os.path.abspath(txt_folder).replace("\\", "/")

                # Parámetros configurables
                KEYPOINT_CONF_THRESH = 0.5
                MAX_FRAMES_SIN_DETECCION = 50
                KPT_HISTORY_SIZE = 10
                #GEOMETRIC_SIMILARITY_THRESH = 0.2
                GEOMETRIC_SIMILARITY_THRESH = 0.35  # Aumentar umbral
                TARGET_FPS = 10
                REID_CONFIRMATION_FRAMES = 3
                HISTORIC_PERSISTENCE = 5

                # Estado del tracking
                tracking_state = {
                    'selected_id': None,
                    'reference_kpts': np.zeros((17, 3)),
                    'last_valid_kpts': np.zeros((17, 3)),
                    'kpts_history': deque(maxlen=KPT_HISTORY_SIZE),
                    'geometric_signature': None,
                    'frames_sin_deteccion': 0, 
                    'signature_history': deque(maxlen=HISTORIC_PERSISTENCE), 
                    'reid_confidence': 0,  # Confirmación de re-identificación
                    'blacklisted_ids': set()  # IDs ignorados
                }

                # Configuración de FPS
                fps = cap.get(cv2.CAP_PROP_FPS)
                print("Configuración de FPS", fps )
                frame_interval = max(1, int(fps // TARGET_FPS))
                frame_count = 0

                # Función para cálculo de firma geométrica
                def calculate_geometric_signature(kpts):
                    valid = kpts[:, 2] > KEYPOINT_CONF_THRESH
                    signature = []                
                    # 1. Anchura de hombros con valor por defecto
                    shoulder_width = 0.0
                    if valid[5] and valid[6]:
                        shoulder_width = np.linalg.norm(kpts[5,:2] - kpts[6,:2])
                    signature.append(shoulder_width)                
                    # 2. Relación hombro-cadera con protección contra división por cero
                    hip_ratio = 0.0
                    if valid[5] and valid[6] and valid[11] and valid[12]:
                        hip_width = np.linalg.norm(kpts[11,:2] - kpts[12,:2])
                        hip_ratio = shoulder_width / hip_width if hip_width != 0 else 0
                    signature.append(hip_ratio)                
                    # 3. Ángulo con valor por defecto
                    angle = 0.0
                    if valid[5] and valid[11] and valid[13]:
                        vec1 = kpts[11] - kpts[5]
                        vec2 = kpts[13] - kpts[11]
                        angle = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
                        angle = np.degrees(angle)
                    signature.append(angle)     
                    torso_length = 0.0
                    if valid[5] and valid[6] and valid[11] and valid[12]:
                        shoulders_center = (kpts[5,:2] + kpts[6,:2])/2
                        hips_center = (kpts[11,:2] + kpts[12,:2])/2
                        torso_length = np.linalg.norm(shoulders_center - hips_center)
                    signature.append(torso_length)
                    # 5. Relación longitud de brazos
                    arm_ratio = 0.0
                    if valid[5] and valid[7] and valid[9] and valid[6] and valid[8] and valid[10]:
                        left_arm = np.linalg.norm(kpts[5,:2]-kpts[7,:2]) + np.linalg.norm(kpts[7,:2]-kpts[9,:2])
                        right_arm = np.linalg.norm(kpts[6,:2]-kpts[8,:2]) + np.linalg.norm(kpts[8,:2]-kpts[10,:2])
                        arm_ratio = left_arm / right_arm if right_arm != 0 else 0
                    signature.append(arm_ratio)            
                    return np.array(signature)  # Ahora siempre retorna 3 elementos

                # Función de similitud geométrica precisa
                def geometric_similarity(ref_sig, cand_sig):
                    if len(tracking_state['signature_history']) == 0:
                        return float('inf')     
                    avg_ref_sig = np.mean(tracking_state['signature_history'], axis=0)           
                    # 2. Asegurar dimensionalidad compatible
                    min_len = min(len(avg_ref_sig), len(cand_sig))
                    ref = avg_ref_sig[:min_len]
                    cand = np.array(cand_sig)[:min_len]
                    
                    # 3. Pesos dinámicos para 5 características (ajustar según tu implementación real)
                    feature_weights = np.array([
                        0.25,  # Ancho hombros
                        0.20,  # Relación hombro-cadera
                        0.15,  # Ángulo postural
                        0.25,  # Longitud torso
                        0.15   # Relación brazos
                    ])[:min_len]
                    
                    # 4. Normalizar pesos
                    feature_weights /= feature_weights.sum() + 1e-6
                    
                    # 5. Cálculo de distancia ponderada con normalización
                    normalized_diff = (ref - cand) / (np.abs(ref) + 1e-6)
                    weighted_distance = np.sqrt(np.sum(feature_weights * normalized_diff**2))
                    
                    return weighted_distance

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    #current_kpts = np.zeros((17, 3))
                    current_kpts = tracking_state['last_valid_kpts'].copy()

                    deteccion_valida = False
                    if frame_count % frame_interval == 0:
                        # Detección principal en intervalos configurados
                        tracking_results = yolo_tracker.track(
                            frame,
                            persist=True,
                            classes=0,
                            conf=0.5,
                            verbose=False,
                            tracker="botsort.yaml"  )
                        boxes = tracking_results[0].boxes.xyxy.cpu().numpy() if tracking_results[0].boxes.id is not None else []
                        track_ids = tracking_results[0].boxes.id.cpu().numpy().astype(int) if tracking_results[0].boxes.id is not None else []       
                        if tracking_state['selected_id'] is None:                        
                            # Selección inicial de persona
                            best_score = -1
                            best_candidate = None                        
                            for i, box in enumerate(boxes):
                                x1, y1, x2, y2 = map(int, box)
                                roi = frame[y1:y2, x1:x2]                            
                                if roi.size == 0:
                                    continue                                
                                pose_results = yolo_pose(roi, conf=0.3, verbose=False)[0]
                                kpts = extract_keypoints(pose_results, box, (x1, y1))                            
                                if np.sum(kpts[:,2] > KEYPOINT_CONF_THRESH) >= 3:  # Mínimo keypoints visibles
                                    signature = calculate_geometric_signature(kpts)
                                    if signature is not None:
                                        area = (x2-x1)*(y2-y1)* 2
                                        score = area + np.sum(kpts[:,2])                                    
                                        if score > best_score:
                                            best_score = score
                                            best_candidate = (i, kpts, signature)                        
                            if best_candidate is not None:
                                i, kpts, signature = best_candidate
                                tracking_state.update({
                                    'selected_id': track_ids[i],
                                    'reference_kpts': kpts.copy(),
                                    'last_valid_kpts': kpts.copy(),
                                    'kpts_history':kpts.copy(),
                                    'geometric_signature': signature,
                                    'frames_sin_deteccion': 0, 
                                    'signature_history': deque(maxlen=10),  # Historial de últimas 10 firmas válidas
                                'reid_confidence': 0,  # Confirmaciones necesarias para cambio de ID
                                'blacklisted_ids': set()  # IDs temporalmente bloqueados

                                    })                            
                                print(f"Tracking iniciado - ID: {track_ids[i]}")
                                deteccion_valida = True
                        else:
                            # Seguimiento activo
                            deteccion_valida = False
                            if tracking_state['selected_id'] in track_ids:
                                idx = np.where(track_ids == tracking_state['selected_id'])[0][0]
                                box = boxes[idx]
                                x1, y1, x2, y2 = map(int, box)
                                roi = frame[y1:y2, x1:x2]                            
                                pose_results = yolo_pose(roi, conf=0.3, verbose=False)[0]
                                new_kpts = extract_keypoints(pose_results, box, (x1, y1))                            
                                if np.sum(new_kpts[:,2] > KEYPOINT_CONF_THRESH) >= 3:
                                    tracking_state.update({
                                        'reference_kpts': new_kpts.copy(),
                                        'last_valid_kpts': new_kpts.copy(),
                                        'kpts_history':new_kpts.copy(),
                                        'frames_sin_deteccion': 0 })
                                deteccion_valida = True
                            else:
                                # Re-identificación geométrica
                                best_match = None
                                min_distance = float('inf')
                                candidate_found = False
                                
                                for i, box in enumerate(boxes):
                                    # Saltar IDs en lista negra
                                    if track_ids[i] in tracking_state['blacklisted_ids']:
                                        continue                                        
                                    x1, y1, x2, y2 = map(int, box)
                                    roi = frame[y1:y2, x1:x2]
                                    
                                    # Extraer keypoints del candidato
                                    pose_results = yolo_pose(roi, conf=0.3, verbose=False)[0]
                                    candidate_kpts = extract_keypoints(pose_results, box, (x1, y1))
                                    
                                    # Validar keypoints del candidato
                                    if candidate_kpts is None or np.sum(candidate_kpts[:,2] > KEYPOINT_CONF_THRESH) < 3:
                                        continue
                                        
                                    # Calcular firma geométrica
                                    candidate_sig = calculate_geometric_signature(candidate_kpts)
                                    
                                    if candidate_sig is not None and len(candidate_sig) == 5:
                                        # Calcular similitud con firma de referencia
                                        distance = geometric_similarity(
                                            tracking_state['geometric_signature'],
                                            candidate_sig
                                        )
                                        
                                        # Actualizar mejor coincidencia
                                        if distance < min_distance and distance < GEOMETRIC_SIMILARITY_THRESH:
                                            min_distance = distance
                                            best_match = (i, candidate_kpts, candidate_sig)
                                            candidate_found = True

                                if best_match is not None and candidate_found:
                                    i, kpts, sig = best_match
                                    
                                    # Requerir confirmación en múltiples frames
                                    tracking_state['reid_confidence'] += 1
                                    
                                    if tracking_state['reid_confidence'] >= REID_CONFIRMATION_FRAMES:
                                        # Actualizar estado con nuevo ID
                                        tracking_state['blacklisted_ids'].add(tracking_state['selected_id'])
                                        
                                        tracking_state.update({
                                            'selected_id': track_ids[i],
                                            'reference_kpts': kpts.copy(),
                                            'last_valid_kpts': kpts.copy(),
                                            'kpts_history': deque([kpts.copy()], maxlen=KPT_HISTORY_SIZE),
                                            'geometric_signature': sig,
                                            'signature_history': deque([sig], maxlen=HISTORIC_PERSISTENCE),
                                            'frames_sin_deteccion': 0,
                                            'reid_confidence': 0
                                        })
                                        deteccion_valida = True
                                        print(f"Re-identificado nuevo ID: {track_ids[i]}")
                                else:
                                    tracking_state['frames_sin_deteccion'] += 1
                                    tracking_state['reid_confidence'] = max(0, tracking_state['reid_confidence'] - 1)
                                    deteccion_valida = False

                            # Resetear tracking si se excede el límite
                            if tracking_state['frames_sin_deteccion'] > MAX_FRAMES_SIN_DETECCION:
                                tracking_state.update({
                                    'selected_id': None,
                                    'reference_kpts': np.zeros((17, 3)),
                                    'last_valid_kpts': np.zeros((17, 3)),
                                    'geometric_signature': None,
                                    'signature_history': deque(maxlen=HISTORIC_PERSISTENCE),
                                    'frames_sin_deteccion': 0,
                                    'reid_confidence': 0
                                })
                                print("Reset por falta de detecciones")
                                deteccion_valida = False

                    # Dibujado preciso usando solo keypoints válidos kpts_history                
                    current_kpts = tracking_state['last_valid_kpts'].copy()
                    file_name = f"frame_{frame_count:06d}"
                    txt_path = os.path.join(txt_folder, f"{file_name}.txt")
                    img_path = os.path.join(images_folder, f"{file_name}.png")
                    #save_keypoints_to_file(txt_path, current_kpts, tracking_state.get('selected_id'))
                    #if deteccion_valida:     
                    # MODIFICACIÓN 3: Validación más estricta para dibujar
                    current_kpts = tracking_state['last_valid_kpts'].copy()
                    has_visible_kpts = np.any(current_kpts[:, 2] > KEYPOINT_CONF_THRESH)
                    
                    # Solo dibujar si hay detección válida Y keypoints visibles
                    #if has_visible_kpts:
                    #if has_visible_kpts and deteccion_valida:
                    if deteccion_valida:
                        frame = draw_pose(frame, current_kpts)
                        save_keypoints_to_file(txt_path, current_kpts, tracking_state.get('selected_id'))
                        cv2.imwrite(img_path, frame)                    
                    frame_count += 1
                cap.release()
                print(f"Proceso completado. Resultados en: {output_folder}")    
                print("txt_folder", txt_folder)  
                
                return archivo_prueba  # Retorna la ruta generada
            except Exception as e:
                print(f"Error en procesamiento: {str(e)}")
                return None 
                
if __name__ == "__main__":

    carpeta_videos = "D:/programs/videos/Videos procesadoss/En movimiento segnmientado Procesado/Otro analisis"
    #carpeta_videos = "D:/programs/videos/otro"
    base_directory = "D:/programs/redNeuronal/modificaciones/datos_agrupar/dataSet/video_04-04"
    print(f"¿Existe la carpeta? {os.path.exists(carpeta_videos)}")
    print(f"Ruta absoluta: {os.path.abspath(carpeta_videos)}")
    print(f"Contenido del directorio padre: {os.listdir(os.path.dirname(carpeta_videos))}")
    FuncionP(carpeta_videos, base_directory)
    