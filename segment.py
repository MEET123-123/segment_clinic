import numpy as np
import cv2
from PIL import Image
import os
import kagglehub
from ultralytics import YOLO

# download datasets
'''
path = kagglehub.dataset_download("balraj98/cvcclinicdb")
print("Path to dataset files:", path)
'''

def transform_mask_txt(mask_path,txt_path):
    # 读取mask文件
    image = cv2.imread(mask_path)
    height, width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    # cv2.imwrite('binary_image.jpg', binary_image)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 打开txt文件准备写入
    with open(txt_path, 'w') as f:
        # only one class
        for contour in contours:
            f.write('0 ')
            # 轮廓近似的精度参数
            epsilon = 0.005 * cv2.arcLength(contour, True)  # 注意：这里使用contour而不是contours[0]
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            # 将轮廓点转换为顶点坐标，并去除重复点
            approx_contour = approx_contour.reshape(-1, 2)
            approx_contour = np.unique(approx_contour, axis=0)
            print('contour.shape=', approx_contour.shape)
            # 将顶点坐标写入txt文件
            for point in approx_contour:
                x, y = point
                x = x / width
                y = y / height
                f.write(f'{x} {y} ')

def convert_tif_to_jpg(tif_file_path, jpg_file_path, quality=95):
    """
    Convert a TIFF file to a JPEG file.

    :param tif_file_path: Path to the TIFF file.
    :param jpg_file_path: Path to the output JPEG file.
    :param quality: JPEG quality, an integer between 1 (worst) and 95 (best). Default is 85.
    """
    try:
        # 打开TIF文件
        with Image.open(tif_file_path) as img:
            # 转换图像到RGB，因为JPG不支持透明通道
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                img = img.convert('RGB')
            # 保存为JPG文件
            img.save(jpg_file_path, 'JPEG', quality=quality)
            print(f"Converted {tif_file_path} to {jpg_file_path} with quality {quality}.")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == '__main__':
    #model = YOLO(r'E:/segment/ultralytics/ultralytics/ultralytics/cfg/models/v8/yolov8-seg.yaml').load("E:/segment/yolov8m-seg.pt")
    #model.train(data=r'E:/segment/ultralytics/ultralytics/ultralytics/cfg/datasets/coco8-seg.yaml',
                #epochs = 100,batch=16,device='cuda')
    model = YOLO(r'E:/segment/pythonProject1/runs/segment/train2/weights/best.pt')
    model.predict(r'E:/segment/val',save=True,boxes=True)
    '''
    for i in range(1,613):
    mask_path = f"E:/segment/process/CVC_ClinicDB/Ground_Truth/{i}.png"
    output_path = f"E:/segment/process/CVC_ClinicDB/labels/{i}.txt"
    transform_mask_txt(mask_path,output_path)
    '''
