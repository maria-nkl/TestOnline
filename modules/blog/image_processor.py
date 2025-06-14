# modules/blog/image_processor.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18
from django.conf import settings
import logging
from typing import Tuple, List, Dict, Optional  # Добавляем необходимые импорты типов

logger = logging.getLogger(__name__)

class PredictionConfig:
    img_size = 64  # Размер изображений для модели

class ImageProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.base_output_dir = os.path.join(settings.MEDIA_ROOT, 'processed_images')
        
        # Создаем необходимые директории
        os.makedirs(self.base_output_dir, exist_ok=True)

    class CustomResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = resnet18(pretrained=False)
            self.resnet.fc = nn.Sequential(
                nn.Linear(self.resnet.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 1)
            )

        def forward(self, x):
            return torch.sigmoid(self.resnet(x))
    
    def _load_model(self):
        """Загружает предварительно обученную модель"""
        model = self.CustomResNet().to(self.device)
        model_path = os.path.join(settings.BASE_DIR, 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def format_processing_results(self, squares_data):
        """
        Возвращает HTML-отчёт с узким столбцом отметок
        """
        result = """
        <div style='font-family: monospace; max-width: 500px;'>
        <h3 style='color: #2c3e50; margin-bottom: 10px;'>📊 Результаты обработки шаблонов</h3>
        """
        
        for template, questions in squares_data.items():
            result += f"""
            <div style='margin-bottom: 25px;'>
                <h4 style='color: #3498db; margin: 5px 0;'>📋 Шаблон: {template}</h4>
                <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
                    <thead>
                        <tr style='background-color: #f8f9fa;'>
                            <th style='padding: 6px; border: 1px solid #dee2e6; text-align: center; width: 30%;'>Вопрос №</th>
                            <th style='padding: 6px; border: 1px solid #dee2e6; text-align: center; width: 30%;'>Отметки</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for question, answers in sorted(questions.items(), key=lambda x: int(x[0])):
                marked = [a for a, s in sorted(answers.items()) if s == "marked"]
                marks = " ".join(f"<span style='color: #28a745;'>✓{a}</span>" for a in marked) if marked else "<span style='color: #dc3545;'>✗</span>"
                
                result += f"""
                    <tr>
                        <td style='padding: 6px; border: 1px solid #dee2e6; text-align: center;'>{question}</td>
                        <td style='padding: 6px; border: 1px solid #dee2e6; text-align: center;'>{marks}</td>
                    </tr>
                """
            
            result += """
                    </tbody>
                </table>
            </div>
            """
        
        result += "</div>"
        return result
    
    def process_uploaded_image(self, image_path: str) -> Dict:
        """
        Основной метод для обработки загруженного изображения.
        Возвращает словарь с результатами обработки или ошибкой.
        """
        try:
            # Создаем уникальные папки для каждого изображения
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            session_dir = os.path.join(self.base_output_dir, base_name)
            
            output_norm_folder = os.path.join(session_dir, 'normalized')
            output_rectangles_folder = os.path.join(session_dir, 'rectangles')
            output_squares_folder = os.path.join(session_dir, 'squares')
            
            os.makedirs(output_norm_folder, exist_ok=True)
            os.makedirs(output_rectangles_folder, exist_ok=True)
            os.makedirs(output_squares_folder, exist_ok=True)

            # 1. Загрузка и предварительная обработка
            image, gray = self._load_and_preprocess_image(image_path)

            # 2. Поиск маркеров
            markers = self._find_markers(gray)
            if len(markers) != 4:
                return {"error": f"Найдено {len(markers)} маркеров (требуется 4)"}

            # 3. Нормализация изображения
            norm_output_path = os.path.join(output_norm_folder, f"{base_name}_norm.jpg")
            normalized = self._normalize_image(image, markers, norm_output_path)
            if normalized is None:
                return {"error": "Невозможно выполнить нормализацию"}

            # 4. Извлечение прямоугольников
            rectangles = self._extract_numbered_rectangles(
                norm_output_path,
                output_rectangles_folder
            )

            # 5. Обработка прямоугольников для поиска квадратов
            squares_data = self._process_squares(
                output_rectangles_folder,
                output_squares_folder
            )

            # Формируем относительные пути для хранения в БД
            rel_path = lambda p: os.path.relpath(p, settings.MEDIA_ROOT)

            
            return self.format_processing_results(squares_data)
        
        except Exception as e:
            logger.error(f"Ошибка обработки изображения {image_path}: {str(e)}")
            return f"## ❌ Ошибка обработки\n\n🛑 Произошла ошибка: {str(e)}"
    

    def _load_and_preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Загружает изображение и преобразует его в оттенки серого."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image, gray

    def _get_marker_corners(self, contour: np.ndarray) -> np.ndarray:
        """Находит 4 угла маркера с помощью аппроксимации контура."""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) != 4:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            approx = np.int0(box)

        approx = approx.reshape(4, 2)
        center = approx.mean(axis=0)
        angles = np.arctan2(approx[:, 1] - center[1], approx[:, 0] - center[0])
        sorted_idx = np.argsort(angles)
        return approx[sorted_idx]

    def _find_markers(self, gray_image: np.ndarray, min_area: int = 500, max_area: int = 5000) -> List[np.ndarray]:
        """Находит угловые маркеры (черные квадраты) на изображении."""
        thresh = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 5
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        markers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                if 0.8 < aspect_ratio < 1.2:
                    corners = self._get_marker_corners(cnt)
                    markers.append(corners)

        return markers

    def _normalize_image(
            self,
            image: np.ndarray,
            markers: List[np.ndarray],
            output_path: str,
            output_size: Tuple[int, int] = (1000, 1500),
            crop_percent: int = 10,
    ) -> Optional[np.ndarray]:
        """Выравнивает изображение и обрезает верх/низ, чтобы удалить маркеры."""
        markers_array = np.array(markers)
        centers = np.array([np.mean(marker, axis=0) for marker in markers_array])

        sorted_indices = np.argsort(centers[:, 1])
        top_indices = sorted_indices[:2]
        bottom_indices = sorted_indices[2:]
        top_sorted = top_indices[np.argsort(centers[top_indices, 0])]
        bottom_sorted = bottom_indices[np.argsort(centers[bottom_indices, 0])[::-1]]
        final_order = np.concatenate([top_sorted, bottom_sorted])
        centers_sorted = centers[final_order]

        src_points = np.float32(centers_sorted)
        width, height = output_size
        dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        result = cv2.warpPerspective(image, matrix, (width, height))

        crop_pixels = int(height * crop_percent / 100)
        result = result[crop_pixels: height - crop_pixels, :]

        cv2.imwrite(output_path, result, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return result

    def _extract_numbered_rectangles(
            self,
            input_image_path: str,
            output_folder: str,
            expected_count: int = 20,
            min_area: int = 20000,
            max_area: int = 50000,
            aspect_ratio_range: Tuple[float, float] = (4.5, 7.5)
    ) -> List[Tuple[int, int, int, int]]:
        """Извлекает прямоугольники с правильной нумерацией для двух столбцов."""
        image = cv2.imread(input_image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {input_image_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 7
        )

        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rectangles = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = max(w, h) / min(w, h)

            if (min_area <= area <= max_area) and (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                rectangles.append((x, y, w, h))

        if len(rectangles) >= 2:
            x_coords = [r[0] for r in rectangles]
            median_x = np.median(x_coords)

            left_col = [r for r in rectangles if r[0] < median_x]
            right_col = [r for r in rectangles if r[0] >= median_x]

            left_col.sort(key=lambda r: r[1])
            right_col.sort(key=lambda r: r[1])

            rectangles = left_col[:10] + right_col[:10]

        base_name = os.path.splitext(os.path.basename(input_image_path))[0]

        for i, (x, y, w, h) in enumerate(rectangles[:expected_count], 1):
            roi = image[y:y + h, x:x + w]
            output_path = os.path.join(output_folder, f"{base_name}_{i}.jpg")
            cv2.imwrite(output_path, roi)

        return rectangles[:expected_count]

    def _find_squares(
            self,
            image: np.ndarray,
            min_area: int = 2000,
            max_area: int = 3000
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Dict]]:
        """Находит квадраты на изображении."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 5
        )

        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        squares = []
        rejected = []

        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            aspect_ratio = float(w) / h
            vertices = len(approx)

            if (vertices == 4 and min_area <= area <= max_area and 
                0.8 <= aspect_ratio <= 1.2 and cv2.isContourConvex(approx)):
                squares.append((x, y, w, h))
            else:
                rejected.append({
                    "position": (x, y, w, h),
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                    "vertices": vertices
                })

        squares.sort(key=lambda s: s[0])
        return squares, rejected

    def _process_squares(self, input_folder: str, output_folder: str) -> Dict[str, Dict[int, Dict[int, str]]]:

        """Обрабатывает все изображения в папке и возвращает словарь с результатами"""
        results = {}
        
        for filename in sorted(os.listdir(input_folder)):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            try:
                # Извлекаем имя шаблона (например, Bushueva из Bushueva_czmgvHD_norm_1_square_2_empty.jpg)
                template_name = filename.split('_')[0]
                if template_name not in results:
                    results[template_name] = {}

                file_path = os.path.join(input_folder, filename)
                image = cv2.imread(file_path)
                if image is None:
                    continue

                squares, _ = self._find_squares(image)
                base_name = os.path.splitext(filename)[0]
                
                for i, (x, y, w, h) in enumerate(squares, 1):
                    square_img = image[y:y + h, x:x + w]
                    square_filename = f"{base_name}_square_{i}.jpg"
                    output_path = os.path.join(output_folder, square_filename)
                    cv2.imwrite(output_path, square_img)

                    # Предсказание
                    prediction = self._predict_square(output_path)
                    status = "empty" if prediction == 0 else "marked"
                    
                    new_filename = f"{base_name}_square_{i}_{status}.jpg"
                    new_path = os.path.join(output_folder, new_filename)
                    os.rename(output_path, new_path)

                    # Извлекаем номер вопроса из имени файла (например, 1 из Bushueva_czmgvHD_norm_1)
                    question_number = int(base_name.split('_')[-1])
                    
                    if question_number not in results[template_name]:
                        results[template_name][question_number] = {}

                    # Сохраняем только marked ответы
                    if status == "marked":
                        results[template_name][question_number][i] = status

            except Exception as e:
                logger.error(f"Ошибка обработки квадратов в {filename}: {str(e)}")

        # Выводим результаты в консоль
        print("\nРезультаты обработки шаблонов:")
        for template, questions in results.items():
            print(f"\nШаблон: {template}")
            for question, answers in sorted(questions.items()):
                print(f"  Вопрос {question}:")
                for answer, status in sorted(answers.items()):
                    print(f"    Вариант {answer}: {status}")
        
        return results


    def _predict_square(self, image_path: str) -> int:
        """Предсказывает состояние квадрата (0 - пустой, 1 - помеченный)"""
        transform = transforms.Compose([
            transforms.Resize((PredictionConfig.img_size, PredictionConfig.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(image)
                return (output > 0.5).float().item()
        except Exception as e:
            logger.error(f"Ошибка предсказания для {image_path}: {str(e)}")
            return 0