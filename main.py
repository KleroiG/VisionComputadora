import cv2
from ultralytics import YOLO
import torch
import pyttsx3
from googletrans import Translator
from collections import defaultdict

# Inicializar síntesis de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Inicializar traductor
translator = Translator()

# Verificar si CUDA está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Cargar modelo YOLO
model = YOLO("yolo11n.pt").to(device)

# Captura desde IP Webcam
url = "http://192.168.26.3:8080/video"
cap = cv2.VideoCapture(url)

# Función para determinar si el objeto está en el centro
def esta_en_centro(x_center, y_center, frame_w, frame_h, margen=0.25):
    centro_x = frame_w / 2
    centro_y = frame_h / 2
    margen_x = frame_w * margen
    margen_y = frame_h * margen
    return (centro_x - margen_x < x_center < centro_x + margen_x) and \
           (centro_y - margen_y < y_center < centro_y + margen_y)

# Traducir nombre del objeto
def traducir_objeto(label):
    try:
        resultado = translator.translate(label, src='en', dest='es')
        return resultado.text.lower()
    except:
        return label

# Decir el texto en voz alta
def hablar(texto):
    engine.say(texto)
    engine.runAndWait()

# Control para evitar repeticiones
objetos_detectados_previamente = set()
frame_counter = 0
skip_frames = 2

# Diccionario de agrupaciones
agrupaciones = {
    "book": "conjunto de libros",
    "pen": "conjunto de lápices",
    "pencil": "conjunto de lápices",
    "bottle": "conjunto de botellas"
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el video.")
        break

    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue

    frame_resized = cv2.resize(frame, (640, 480))
    frame_w, frame_h = frame_resized.shape[1], frame_resized.shape[0]
    results = model(frame_resized)[0]

    objetos_visibles_ahora = set()
    agrupados = defaultdict(list)

    for box in results.boxes:
        label = model.names[int(box.cls)]
        conf = box.conf.item()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        if not esta_en_centro(x_center, y_center, frame_w, frame_h):
            continue

        # Agrupar si corresponde
        if label in agrupaciones:
            agrupados[agrupaciones[label]].append((x1, y1, x2, y2))
        else:
            objetos_visibles_ahora.add(label)

    # Anunciar agrupaciones
    for grupo in agrupados:
        if grupo not in objetos_detectados_previamente:
            hablar(f"Estas observando un {grupo}")
            objetos_detectados_previamente.add(grupo)

    # Anunciar objetos individuales
    for obj in objetos_visibles_ahora:
        if obj not in objetos_detectados_previamente:
            nombre = traducir_objeto(obj)
            articulo = "una" if nombre.endswith("a") else "un"
            hablar(f"Estas observando {articulo} {nombre}")
            objetos_detectados_previamente.add(obj)

    # Limpiar objetos que ya no están
    objetos_detectados_previamente = {
        obj for obj in objetos_detectados_previamente
        if obj in objetos_visibles_ahora or obj in agrupados
    }

    # Mostrar el frame (opcional para depuración)
    cv2.imshow("Detección", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
