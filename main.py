import cv2
from ultralytics import YOLO
import torch

# Verificar si CUDA está disponible
if torch.cuda.is_available():
    device = "cuda"  # Usar GPU
else:
    device = "cpu"   # Usar CPU

print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

# Cargar el modelo YOLO
model = YOLO("yolo11n.pt")

# Mover el modelo a la GPU si está disponible
model.to(device)

# URL del stream de la cámara (usando IP Webcam)
url = "http://192.168.26.5:8080/video"

# Capturar el video
cap = cv2.VideoCapture(url)

# Función para determinar la ubicación del objeto
def get_object_location(x_center, frame_width):
    if x_center < frame_width / 3:
        return "izquierda"
    elif x_center > 2 * frame_width / 3:
        return "derecha"
    else:
        return "centro"

# Bucle principal
frame_counter = 0
skip_frames = 2  # Procesar 1 de cada 2 frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el video.")
        break

    frame_counter += 1
    if frame_counter % skip_frames != 0:
        continue  # Saltar este frame

    # Reducir la resolución del frame para mejorar el rendimiento
    resized_frame = cv2.resize(frame, (640, 480))
    frame_width = resized_frame.shape[1]

    # Detectar objetos en el frame
    results = model(resized_frame)

    # Procesar los resultados
    for result in results:
        boxes = result.boxes  # Obtener las cajas delimitadoras
        for box in boxes:
            label = model.names[int(box.cls)]  # Obtener la etiqueta de la clase
            confidence = box.conf.item()  # Obtener la confianza
            x_center = (box.xyxy[0][0] + box.xyxy[0][2]) / 2  # Calcular el centro en el eje X
            location = get_object_location(x_center, frame_width)  # Obtener la ubicación

            # Mostrar información en la consola
            print(f"Detected: {label} ({confidence:.2f}) - Ubicación: {location}")

            # Dibujar la caja delimitadora y la ubicación en el frame
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(resized_frame, f"{label} ({location})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Mostrar el frame (opcional, para depuración)
    cv2.imshow('YOLOv11 Object Detection', resized_frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()