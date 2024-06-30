import cv2
import dlib
import numpy as np

# 顔検出モデルとランドマーク予測モデルの読み込み
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 画像の読み込み
image = cv2.imread("Lenna.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 顔の検出
faces = detector(gray)
for face in faces:
    landmarks = predictor(gray, face)
    
    # 目のランドマークを取得（左目：36-41、右目：42-47）
    left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
    
    # 口のランドマークを取得（48-67）
    mouth_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])
    
    # 目の領域の抽出
    left_eye_hull = cv2.convexHull(left_eye_pts)
    right_eye_hull = cv2.convexHull(right_eye_pts)
    cv2.drawContours(image, [left_eye_hull], -1, (0, 255, 0), 1)
    cv2.drawContours(image, [right_eye_hull], -1, (0, 255, 0), 1)
    
    # 口の領域の抽出
    mouth_hull = cv2.convexHull(mouth_pts)
    cv2.drawContours(image, [mouth_hull], -1, (0, 0, 255), 1)

    # 簡単な変形（拡大）
    left_eye_center = left_eye_pts.mean(axis=0).astype("int")
    right_eye_center = right_eye_pts.mean(axis=0).astype("int")
    mouth_center = mouth_pts.mean(axis=0).astype("int")

    # 左目を大きくする
    for i in range(36, 42):
        landmarks.part(i).x = int((landmarks.part(i).x - left_eye_center[0]) * 1.2 + left_eye_center[0])
        landmarks.part(i).y = int((landmarks.part(i).y - left_eye_center[1]) * 1.2 + left_eye_center[1])

    # 右目を大きくする
    for i in range(42, 48):
        landmarks.part(i).x = int((landmarks.part(i).x - right_eye_center[0]) * 1.2 + right_eye_center[0])
        landmarks.part(i).y = int((landmarks.part(i).y - right_eye_center[1]) * 1.2 + right_eye_center[1])

    # 口を大きくする
    for i in range(48, 68):
        landmarks.part(i).x = int((landmarks.part(i).x - mouth_center[0]) * 1.2 + mouth_center[0])
        landmarks.part(i).y = int((landmarks.part(i).y - mouth_center[1]) * 1.2 + mouth_center[1])

    # 結果の表示
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(image, (x, y), 2, (255, 0, 0), -1)

# 結果画像の保存
output_path = 'transformed_face.png'
cv2.imwrite(output_path, image)

cv2.imshow("Transformed Face", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
