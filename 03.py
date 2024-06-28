import cv2
import boto3
import numpy as np
import base64
from botocore.config import Config
from matplotlib import pyplot as plt

# AWSの設定
# 注意: 環境変数やAWS IAMロールを使用することを強く推奨します
aws_access_key_id = ''
aws_secret_access_key = ''
region_name = 'ap-northeast-1'

# Rekognitionクライアントの設定
config = Config(retries={'max_attempts': 10, 'mode': 'standard'})

def base64_to_cv2(image_base64):
    image_bytes = base64.b64decode(image_base64)
    np_array = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

def cv2_to_base64(image_cv2):
    _, buffer = cv2.imencode('.jpg', image_cv2)
    return base64.b64encode(buffer).decode()

def rekog_eye(im):
    client = boto3.client('rekognition', region_name, 
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key,
                          config=config)
    
    _, buf = cv2.imencode('.jpg', im)
    faces = client.detect_faces(Image={'Bytes':buf.tobytes()}, Attributes=['ALL'])
    
    landmarks = faces['FaceDetails'][0]['Landmarks']
    eye_types = ['leftEyeLeft', 'leftEyeRight', 'leftEyeUp', 'leftEyeDown', 
                 'rightEyeLeft', 'rightEyeRight', 'rightEyeUp', 'rightEyeDown']
    
    h, w, _ = im.shape
    return {landmark['Type']: {'X': int(landmark['X'] * w), 'Y': int(landmark['Y'] * h)}
            for landmark in landmarks if landmark['Type'] in eye_types}

def mosaic_area(src, x, y, width, height, blur_num):
    dst = src.copy()
    for _ in range(blur_num):
        dst[y:y + height, x:x + width] = cv2.GaussianBlur(dst[y:y + height, x:x + width], (3,3), 3)
    return dst

def process_image(im, magnification, blur_num):
    EyePoints = rekog_eye(im)
    bityouseix, bityouseiy = 20, 5

    leftTop = min(EyePoints[key]['Y'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft'])
    leftBottom = max(EyePoints[key]['Y'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft'])
    leftRight = max(EyePoints[key]['X'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft'])
    leftLeft = min(EyePoints[key]['X'] for key in ['leftEyeUp', 'leftEyeDown', 'leftEyeRight', 'leftEyeLeft'])
    
    rightTop = min(EyePoints[key]['Y'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft'])
    rightBottom = max(EyePoints[key]['Y'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft'])
    rightRight = max(EyePoints[key]['X'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft'])
    rightLeft = min(EyePoints[key]['X'] for key in ['rightEyeUp', 'rightEyeDown', 'rightEyeRight', 'rightEyeLeft'])

    leftEye = im[leftTop:leftBottom+bityouseiy, leftLeft-bityouseix:leftRight+bityouseix]
    leftEye = cv2.resize(leftEye, (leftEye.shape[1], int(leftEye.shape[0]*magnification)))
    rightEye = im[rightTop:rightBottom+bityouseiy, rightLeft-bityouseix:rightRight+bityouseix]
    rightEye = cv2.resize(rightEye, (rightEye.shape[1], int(rightEye.shape[0]*magnification)))

    im[leftTop:leftTop+leftEye.shape[0], leftLeft-bityouseix:leftLeft+leftEye.shape[1]-bityouseix] = leftEye
    im[rightTop:rightTop+rightEye.shape[0], rightLeft-bityouseix:rightLeft+rightEye.shape[1]-bityouseix] = rightEye

    # Apply blurring
    blur_areas = [
        (leftLeft-bityouseix-int(bityouseix/2), leftTop, bityouseix, leftEye.shape[0]+bityouseiy),
        (leftRight+int(bityouseix/2), leftTop, bityouseix, leftEye.shape[0]+bityouseiy),
        (leftLeft-bityouseix, leftTop+leftEye.shape[0]-int(bityouseiy/2), leftEye.shape[1], bityouseiy),
        (rightLeft-bityouseix-int(bityouseix/2), rightTop, bityouseix, rightEye.shape[0]+bityouseiy),
        (rightRight+int(bityouseix/2), rightTop, bityouseix, rightEye.shape[0]+bityouseiy),
        (rightLeft-bityouseix, rightTop+rightEye.shape[0]-int(bityouseiy/2), rightEye.shape[1], bityouseiy)
    ]

    for area in blur_areas:
        im = mosaic_area(im, *area, blur_num)

    return im

def handler(event, context):
    try:
        base_64ed_image = event.get('myimg', 'none')
        magnification = float(event.get('magni', 1.4))
        blur_num = int(event.get('blur', 3))
        im = base64_to_cv2(base_64ed_image)
        processed_im = process_image(im, magnification, blur_num)
        return {'status': 200, 'message': 'OK', 'img': cv2_to_base64(processed_im)}
    except Exception as e:
        return {'status': 500, 'message': str(e)}

def display_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path = '640.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read the image file: {image_path}")
    else:
        print("Original Image:")
        display_image(image)
        
        base64_image = cv2_to_base64(image)
        event = {'myimg': base64_image, 'magni': 1.4, 'blur': 3}
        result = handler(event, None)
        
        if result['status'] == 200:
            processed_image = base64_to_cv2(result['img'])
            print("Processed Image:")
            display_image(processed_image)
        else:
            print(f"Error: {result['message']}")