# OVision-test
Тестовое задание на вакансию Python Developer

# Setup
Для того чтобы использовать класс LandmarkDetector необходимо установить следующие пакеты:<br/>
библиотеку opencv, команда<br/>
    sudo apt-get install python3-opencv<br/>
фреймворк для unit тестов, команда:<br/>
    sudo pip install pytest<br/>
  
# Using
Чтобы найти landmarks на изображении необходимо использовать следующую команду<br/>
python demo.py<br/>
В файле demo.py необходимо прописать следующее<br/>
  model_path = './snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar' - путь где находится тренированная модель<br/>
  image_path = './cache_data/cache/test_1.jpg'                                                - путь до картинки, для которой мы хотим найти landmarks<br/>
  image_result_path = './cache_data/cache/test_1_result.png'                                  - путь где сохранить результат - исходную картинку с landmarks<br/>
  face_position = [819.27, 432.15, 971.70, 575.87]                                            - координаты лица на картинки, для которой ищем landmarks<br/>
  
Результатом будем картинка с landmarks и словарь с полями landmarks (numpy array размера N x 3 - координаты x, y<br/>
и вероятность p для каждой из N точек), где N - число найденных landmarks<br/>

# Unit test
команда  pytest test_landmark_detector.py запускает тестирование методов класса LandmarkDetector:<br/>
  test_predict - тестирует размер тензора предобработанного изображения<br/>
  test_preprocess_image - тестирует предсказанные точки landmarks, округленные до целых чисел

