# OVision-test
Тестовое задание на вакансию Python Developer

# Setup
Для того чтобы использовать класс LandmarkDetector необходимо установить следующие пакеты:
библиотеку opencv, команда
  sudo apt-get install python3-opencv
фреймворк для unit тестов, команда:
  sudo pip install pytest
  
# Using
Чтобы найти landmarks на изображении необходимо использовать следующую команду
python demo.py
В файле demo.py необходимо прописать следующее
  model_path = './snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar' - путь где находится тренированная модель
  image_path = './cache_data/cache/test_1.jpg'                                                - путь до картинки, для которой мы хотим найти landmarks
  image_result_path = './cache_data/cache/test_1_result.png'                                  - путь где сохранить результат - исходную картинку с landmarks
  face_position = [819.27, 432.15, 971.70, 575.87]                                            - координаты лица на картинки, для которой ищем landmarks
  
Результатом будем картинка с landmarks и словарь с полями landmarks (numpy array размера N x 3 -
координаты x, y и вероятность p для каждой из N точек), где N - число найденных landmarks

# Unit test
команда  pytest test_landmark_detector.py запускает тестирование методов класса LandmarkDetector:
  test_predict - тестирует размер тензора предобработанного изображения
  test_preprocess_image - тестирует предсказанные точки landmarks, округленные до целых чисел

