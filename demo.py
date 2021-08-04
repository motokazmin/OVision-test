from landmark_detector import LandmarkDetector
from visualization import draw_image_by_points
import cv2

model_path = './snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar'
image_path = './cache_data/cache/test_1.jpg'
image_result_path = './cache_data/cache/test_1_result.png'
face_position = [819.27, 432.15, 971.70, 575.87]

detector = LandmarkDetector(model_path)
detector.preprocess_image(image_path, face_position)
landmarks, error_message = detector.predict()

image = draw_image_by_points(image_path, detector.prediction, 1, (255,255,255), False)
cv2.imwrite(image_result_path, image)


print(landmarks)


