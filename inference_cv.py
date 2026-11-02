import cv2
import numpy as np
import sys
import threading
import time
import tensorflow as tf

from mqtt import connect_mqtt, send_mqtt, disconnect_mqtt
from model import build_model

latest_frame = None
latest_boxes = []
frame_lock = threading.Lock()
running = True


def rtsp_reader(rtsp_stream):
    global latest_frame, running

    # Abertura do fluxo RTSP através do backend FFmpeg
    cap = cv2.VideoCapture(rtsp_stream, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Erro ao abrir o fluxo RTSP")
        running = False
        return

    while running:
        # Descarte do frame antigo armazenado no buffer
        cap.grab()  # Descarta o frame antigo do buffer
        ret, frame = cap.retrieve()

        if not ret:
            continue

        with frame_lock:
            latest_frame = frame

    cap.release()


def video_inference(fire_model, smoke_model):
    global latest_frame, latest_boxes, running

    # Conversão do modelo Keras numa função TensorFlow
    infer_fire = tf.function(fire_model)
    prev_time = time.time()
    last_infer_time = 0.0
    inference_interval = 0.3  # Intervalo mínimo entre inferências

    fire_prob = 0.0
    smoke_conf = 0.0
    warning_level = 1

    while running:
        # Obtenção segura do frame e das caixas mais recentes
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()
            boxes = list(latest_boxes)

        # Preparação do frame utilizado na apresentação
        display_frame = frame.copy()

        # Cálculo da taxa de frames apresentada
        current_time = time.time()
        elapsed = max(current_time - prev_time, 1e-9)
        fps = 1 / elapsed

        # Execução periódica dos modelos de inferência
        if current_time - last_infer_time >= inference_interval:
            start_time = time.perf_counter()

            # Preparação das imagens de entrada dos dois modelos
            smoke_frame = cv2.resize(frame, (512, 512))
            fire_frame = cv2.resize(frame, (384, 384))

            # Deteção de fumo com o modelo YOLO
            smoke_result = smoke_model.predict(
                smoke_frame,
                imgsz=512,
                verbose=False,
            )[0]

            new_boxes = []
            if len(smoke_result.boxes) > 0:
                height, width = smoke_frame.shape[:2]
                scale_x = frame.shape[1] / width
                scale_y = frame.shape[0] / height

                for box in smoke_result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    new_boxes.append((
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                        int(x2 * scale_x),
                        int(y2 * scale_y),
                    ))

                smoke_conf = float(smoke_result.boxes.conf.max())
            else:
                smoke_conf = 0.0

            # Atualização segura das caixas detetadas
            with frame_lock:
                latest_boxes = new_boxes
                boxes = list(latest_boxes)

            # Classificação da presença de fogo
            fire_input = fire_frame.astype(np.float32)[None, ...]
            prediction = infer_fire(fire_input, training=False)
            fire_prob = float(prediction[0][0])

            print(f"Confiança de fumo: {smoke_conf:.3f}")
            print(f"Probabilidade de fogo: {fire_prob:.3f}")

            end_time = time.perf_counter()
            print(f"Inferência total: {(end_time - start_time) * 1000:.1f} ms")

            warning_level = compute_warning(smoke_conf, fire_prob)
            send_mqtt(warning_level, smoke_conf, fire_prob)
            last_infer_time = current_time

        # Desenho das caixas delimitadoras
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                4,
            )

        # Sobreposição dos resultados no vídeo
        cv2.putText(
            display_frame,
            f"Probabilidade de fogo: {fire_prob:.3f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"Confiança de fumo: {smoke_conf:.3f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            display_frame,
            f"Nível de alerta: {warning_level}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("frame", display_frame)
        prev_time = current_time

        if cv2.waitKey(1) == ord("q"):
            running = False
            break

    cv2.destroyAllWindows()


def compute_warning(smoke_conf, fire_prob):
    """Determina o nível de alerta a partir das previsões dos modelos."""
    # Nível 4 - crítico
    if smoke_conf >= 0.65 or fire_prob >= 0.95:
        return 4

    # Nível 3 - risco elevado
    if (
        0.35 <= smoke_conf < 0.65
        and 0.65 <= fire_prob < 0.95
    ):
        return 3

    # Nível 2 - aviso
    if (
        0.35 <= smoke_conf < 0.65
        or 0.15 <= fire_prob < 0.65
    ):
        return 2

    # Nível 1 - normal
    return 1

if __name__ == "__main__":
    # Validação do argumento com o endereço do fluxo RTSP
    if len(sys.argv) < 2:
        print(f"Utilização: python {sys.argv[0]} <rtsp_url>")
        sys.exit(1)

    stream = sys.argv[1]

    # Configuração do TensorFlow e da GPU
    tf.keras.backend.clear_session()
    gpus = tf.config.list_physical_devices("GPU")

    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    tf.config.optimizer.set_jit(False)

    # Ligação ao broker MQTT e carregamento dos modelos
    connect_mqtt()

    fire_model, smoke_model = build_model()

    # Thread responsável pela leitura do fluxo RTSP
    reader_thread = threading.Thread(
        target=rtsp_reader,
        args=(stream,),
    )
    # Thread responsável pela inferência e apresentação
    inference_thread = threading.Thread(
        target=video_inference,
        args=(fire_model, smoke_model),
    )

    reader_thread.start()
    inference_thread.start()
    inference_thread.join()
    reader_thread.join()

    # Encerramento da comunicação MQTT
    disconnect_mqtt()
