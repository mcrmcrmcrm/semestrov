import cv2
from ultralytics import YOLO

# 1. Инициализация модели YOLO (автоматически скачает pretrained модель)
model = YOLO('yolov8n.pt')  # Можно использовать 'yolov8s.pt' для большей точности

# 2. Подключение веб-камеры
cap = cv2.VideoCapture(0)  # 0 - встроенная камера

# 3. Основной цикл обработки видеопотока
while cap.isOpened():
    # Захват кадра
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция объектов
    results = model(frame, verbose=False)  # Убираем лишний вывод

    # Отрисовка результатов
    for result in results:
        for box in result.boxes:
            # Получаем данные детекции
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты прямоугольника
            class_id = int(box.cls)  # ID класса
            confidence = float(box.conf)  # Уверенность (0-1)
            class_name = model.names[class_id]  # Название класса

            # Отрисовка прямоугольника
            color = (0, 255, 0)  # Зеленый цвет (BGR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Отрисовка подписи (класс + вероятность)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Вывод кадра
    cv2.imshow('YOLO Object Detection', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()