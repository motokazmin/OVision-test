from landmark_detector import LandmarkDetector

model_path = './snapshots/SAN_300W_GTB_itn_cpm_3_50_sigma4_128x128x8/checkpoint_49.pth.tar'
image_path = './cache_data/cache/test_1.jpg'
image_result_path = './cache_data/cache/test_1_result.png'
face_position = [819.27, 432.15, 971.70, 575.87]

detector = LandmarkDetector(model_path)

detector.preprocess_image(image_path, face_position)

landmarks, error_message = detector.predict()

detector.save_image_with_points(image_result_path)

print(f'\npredicted landmarks:\n {landmarks}')
