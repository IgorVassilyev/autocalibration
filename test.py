import cv2
import os
import glob

# Папки
INPUT_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Используем словарь ArUco 4x4 (можно менять на 4X4_100/250/1000 при необходимости)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
parameters = cv2.aruco.DetectorParameters()

# Собираем список изображений
images = []
for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
    images.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

print(f"Найдено {len(images)} изображений")

for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Не удалось прочитать {img_path}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Поиск маркеров
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    # Отрисовка
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        print(f"[OK] {img_path}: найдено {len(ids)} маркер(ов)")
    else:
        print(f"[..] {img_path}: маркеры не найдены")

    # Сохраняем результат
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cv2.imwrite(out_path, img)

print("Готово! Результаты сохранены в папку", OUTPUT_DIR)
