from landmark_detector import LandmarkDetector
import numpy as np

model_path = './snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar'
image_path = './cache_data/cache/test_1.jpg'
image_result_path = './cache_data/cache/test_1_result.png'
face_position = [819.27, 432.15, 971.70, 575.87]

def test_predict():
    detector = LandmarkDetector(model_path)
    detector.preprocess_image(image_path, face_position)
    landmarks, error_message = detector.predict()

    points = np.array([int(item) for t in landmarks for item in t])
    true_points = [832, 441, 850, 438, 867, 444, 901, 444, 923, 440, 942, 444, 841, 457, 854, 457, 867, 461, 904, 462, 920, 459, 933, 460, 863, 494, 880, 493, 902, 497, 858, 518, 883, 526, 911, 521, 882, 562]

    assert np.sum(points == true_points) == len(true_points)

def test_preprocess_image():
    detector = LandmarkDetector(model_path)
    detector.preprocess_image(image_path, face_position)
    landmarks, error_message = detector.predict()

    assert list(detector.inputs.shape) == [1, 3, 128, 128]
