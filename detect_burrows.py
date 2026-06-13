#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Распознавание нор в полях на основе анализа изображений.
Используется преобразование Хафа для поиска кругов.
"""

import argparse
import cv2
import numpy as np
import sys

def detect_burrows(image_path, output_path=None, min_radius=5, max_radius=30,
                   param1=50, param2=30):
    """
    Детектирует норы как круги на изображении.

    Параметры:
        image_path (str): путь к входному изображению
        output_path (str): путь для сохранения результата (опционально)
        min_radius (int): минимальный радиус норы в пикселях
        max_radius (int): максимальный радиус норы в пикселях
        param1 (int): верхний порог Canny для метода Хафа
        param2 (int): порог аккумулятора для детекции кругов (меньше = больше ложных)

    Возвращает:
        list: список найденных кругов (x, y, радиус)
    """
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось загрузить изображение из {image_path}")
        sys.exit(1)

    # Копия для отрисовки результата
    output_img = img.copy()

    # Переводим в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Применяем размытие для уменьшения шума
    blurred = cv2.medianBlur(gray, 5)

    # Детекция кругов методом Хафа
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=10, param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)

    detected_burrows = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Рисуем круг и центр
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output_img, (x, y), 2, (0, 0, 255), 3)
            detected_burrows.append((x, y, r))
            print(f"Найдена нора: центр=({x},{y}) радиус={r}")

    # Сохраняем результат, если указан путь
    if output_path:
        cv2.imwrite(output_path, output_img)
        print(f"Результат сохранён в {output_path}")
    else:
        # Показываем окно с результатом
        cv2.imshow("Обнаруженные норы", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detected_burrows

def main():
    parser = argparse.ArgumentParser(description="Распознавание нор в полях на изображении")
    parser.add_argument("input", help="Путь к входному изображению")
    parser.add_argument("-o", "--output", help="Путь для сохранения результата (опционально)")
    parser.add_argument("--min-radius", type=int, default=5, help="Минимальный радиус норы (пиксели)")
    parser.add_argument("--max-radius", type=int, default=30, help="Максимальный радиус норы (пиксели)")
    parser.add_argument("--param1", type=int, default=50, help="Верхний порог Canny (обычно 30-100)")
    parser.add_argument("--param2", type=int, default=30, help="Порог аккумулятора (меньше = больше кругов)")

    args = parser.parse_args()

    print(f"Обработка изображения: {args.input}")
    detect_burrows(args.input, args.output, args.min_radius, args.max_radius,
                   args.param1, args.param2)

if __name__ == "__main__":
    main()