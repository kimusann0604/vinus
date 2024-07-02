import streamlit as st
import numpy as np
import cv2
import math
from PIL import Image
import pandas as pd
import base64
import boto3
from botocore.config import Config
from io import BytesIO  

# AWSの設定
# 注意: 環境変数やAWS IAMロールを使用することを強く推奨します
aws_access_key_id = ''
aws_secret_access_key = ''
region_name = 'ap-northeast-1'

image_url = "image.jpg"

# 歪み関数
def apply_distortion(im_cv, scale_x, scale_y, amount):
    (h, w, _) = im_cv.shape
    flex_x = np.zeros((h, w), np.float32)
    flex_y = np.zeros((h, w), np.float32)
    center_x, center_y = w / 2, h / 2
    radius = min(center_x, center_y)

    for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y
            if distance >= (radius * radius):
                flex_x[y, x], flex_y[y, x] = x, y
            else:
                factor = 1.0 if distance == 0.0 else math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), -amount)
                flex_x[y, x] = factor * delta_x / scale_x + center_x
                flex_y[y, x] = factor * delta_y / scale_y + center_y

    return cv2.remap(im_cv, flex_x, flex_y, cv2.INTER_LINEAR)

# Rekognition関連の関数
def rekog_eye(im):
    client = boto3.client('rekognition', region_name, 
                          aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key,
                          config=Config(retries={'max_attempts': 10, 'mode': 'standard'}))
    
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

# データフレームの作成
data = {
    'クリニック名': ['高須クリニック', '品川美容外科', 'TCB東京中央美容外科', '湘南美容外科'],
    '評価': [3.4, 3.7, 4.0, 4.6]
}
df = pd.DataFrame(data)

# Streamlit UI
st.sidebar.title("アプリ選択")
page = st.sidebar.radio("", ("カメラ&顔の輪郭加工", "目の加工", "美容外科ランキング"))

if page == "カメラ&顔の輪郭加工":
    st.title("画像歪み調整ツール＆カメラで写真を撮るアプリ")

    image_source = st.radio("画像ソースを選択", ("ファイルをアップロード", "カメラを使用"))

    if image_source == "ファイルをアップロード":
        uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="アップロードされた画像", use_column_width=True)
    else:
        try:
            picture = st.camera_input("写真を撮る")
            if picture:
                image = Image.open(picture)
                st.image(image, caption="撮影された画像", use_column_width=True)
        except Exception as e:
            st.error(f"カメラの使用中にエラーが発生しました: {str(e)}")
            st.info("カメラが利用できない場合は、ファイルのアップロードをお試しください。")

    if 'image' in locals():
        im_cv = np.array(image)

        scale_x = st.slider("スケールX", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        scale_y = st.slider("スケールY", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        amount = st.slider("変化させる量", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

        distorted_image = apply_distortion(im_cv, scale_x, scale_y, amount)
        distorted_image = Image.fromarray(distorted_image)

        st.image([image, distorted_image], caption=['元の画像', '歪んだ画像'], width=300)

        if st.button("元の画像を保存"):
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:file/png;base64,{img_str}" download="original_image.png">元の画像をダウンロード</a>'
            st.markdown(href, unsafe_allow_html=True)

        if st.button("画像を保存"):
            buffered = BytesIO()
            distorted_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            href = f'<a href="data:file/png;base64,{img_str}" download="distorted_image.png">画像をダウンロード</a>'
            st.markdown(href, unsafe_allow_html=True)

elif page == "目の加工":
    st.title("Eye Magnification App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    magnification = st.slider("Magnification", min_value=1.0, max_value=2.0, value=1.4, step=0.1)
    blur_num = 1

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Original Image", use_column_width=True)

        if st.button("Process Image"):
            try:
                processed_image = process_image(image, magnification, blur_num)
                processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                st.image(processed_image_rgb, caption="Processed Image", use_column_width=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

elif page == "美容外科ランキング":
    import streamlit as st

    st.title("美容外科ランキング")

    # メイン目次
    with st.expander("目次", expanded=True):
        st.markdown("""
        1. [美容外科・美容皮膚科人気おすすめクリニック一覧表](#section-1)
        2. [美容外科・美容皮膚科で行う美容整形とは](#section-2)
        3. [美容整形クリニックの選び方・ポイント](#section-3)
        4. [美容整形おすすめ診療メニュー【施術別】](#section-4)
        5. [美容整形クリニックおすすめ人気ランキングTOP10](#section-5)
        """)

    # サブセクション
    with st.expander("3. 美容整形クリニックの選び方・ポイント", expanded=False):
        st.markdown("""
        - 3-1. 安いだけでクリニックを選ばない
        - 3-2. 自分が希望する施術を得意とする医師が在籍しているか
        - 3-3. カウンセリングや治療の説明が丁寧か
        - 3-4. アフターケアや保証制度がしっかりしているか
        """)

    with st.expander("4. 美容整形おすすめ診療メニュー【施術別】", expanded=False):
        st.markdown("""
        - 4-1. 二重整形
        - 4-2. 鼻整形
        - 4-3. 唇プチ整形
        - 4-4. 輪郭形成
        - 4-5. 豊胸
        - 4-6. ボディーライン
        """)

    # 各セクションのダミーコンテンツ
    st.header("1. 美容外科・美容皮膚科人気おすすめクリニック一覧表")
    st.write("ここにクリニック一覧表のコンテンツが入ります。")

    st.header("2. 美容外科・美容皮膚科で行う美容整形とは")
    st.write("美容整形の説明がここに入ります。")

    # 以下、各セクションのコンテンツを追加...

    # カスタムCSSを適用してスタイリングを調整
    st.markdown("""
    <style>
    .stExpander {
        border: 1px solid #e6e6e6;
        border-radius: 4px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stExpander > div > p {
        font-size: 16px;
        margin-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)