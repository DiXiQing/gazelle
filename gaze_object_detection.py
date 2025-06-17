import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from retinaface import RetinaFace
from ultralytics import YOLO
import cv2

def load_models():
    """Load both Gaze-LLE and YOLO models"""
    # Load Gaze-LLE model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gaze_model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
    gaze_model.eval()
    gaze_model.to(device)
    
    # Load YOLO model
    yolo_model = YOLO('yolov8n.pt')  # 使用YOLOv8n作为基础模型
    
    return gaze_model, transform, yolo_model, device

def load_image(image_path):
    """Load image from local path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    return Image.open(image_path)

def detect_faces(image):
    """Detect faces in the image using RetinaFace"""
    resp = RetinaFace.detect_faces(np.array(image))
    bboxes = [resp[key]['facial_area'] for key in resp.keys()]
    return bboxes

def detect_objects(image, yolo_model):
    """Detect objects in the image using YOLO"""
    results = yolo_model(image)
    return results[0].boxes.data.cpu().numpy()  # 返回检测到的物体边界框

def get_gaze_target(heatmap, image_size):
    """Get the gaze target coordinates from heatmap"""
    heatmap_np = heatmap.detach().cpu().numpy()
    max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
    gaze_target_x = max_index[1] / heatmap_np.shape[1] * image_size[0]
    gaze_target_y = max_index[0] / heatmap_np.shape[0] * image_size[1]
    return (gaze_target_x, gaze_target_y)

def find_gazed_object(gaze_point, object_boxes, image_size):
    """Find which object is being gazed at"""
    gaze_x, gaze_y = gaze_point
    max_iou = 0
    gazed_object = None
    
    for box in object_boxes:
        x1, y1, x2, y2, conf, cls = box
        # 计算IOU
        box_area = (x2 - x1) * (y2 - y1)
        gaze_area = 1  # 假设注视点是一个1x1的区域
        intersection_area = max(0, min(x2, gaze_x + 0.5) - max(x1, gaze_x - 0.5)) * \
                          max(0, min(y2, gaze_y + 0.5) - max(y1, gaze_y - 0.5))
        iou = intersection_area / (box_area + gaze_area - intersection_area)
        
        if iou > max_iou:
            max_iou = iou
            gazed_object = (box, iou)
    
    return gazed_object

def visualize_results(pil_image, gaze_points, object_boxes, gazed_objects):
    """Visualize gaze points and detected objects"""
    overlay_image = pil_image.copy()
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size
    
    # 绘制所有检测到的物体
    for box in object_boxes:
        x1, y1, x2, y2, conf, cls = box
        draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
        draw.text((x1, y1-10), f"{yolo_model.names[int(cls)]} {conf:.2f}", fill="blue")
    
    # 绘制注视点和注视的物体
    for i, (gaze_point, gazed_object) in enumerate(zip(gaze_points, gazed_objects)):
        if gazed_object:
            box, iou = gazed_object
            x1, y1, x2, y2, conf, cls = box
            # 绘制注视点
            draw.ellipse([(gaze_point[0]-5, gaze_point[1]-5), 
                         (gaze_point[0]+5, gaze_point[1]+5)], 
                        fill="red")
            # 高亮显示被注视的物体
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1-10), 
                     f"Gazed: {yolo_model.names[int(cls)]} (IoU: {iou:.2f})", 
                     fill="red")
    
    return overlay_image

def main():
    # 设置图片路径和输出目录
    image_path = "testimg_gaze_4.png"  # 在这里修改你的图片路径
    save_dir = "output"
    
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    gaze_model, transform, yolo_model, device = load_models()
    
    # 加载图片
    image = load_image(image_path)
    image = image.convert('RGB')
    width, height = image.size

    # 检测人脸
    bboxes = detect_faces(image)
    if not bboxes:
        print("No faces detected in the image!")
        return

    # 准备Gaze-LLE模型输入
    img_tensor = transform(image).unsqueeze(0).to(device)
    norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]

    input = {
        "images": img_tensor,
        "bboxes": norm_bboxes
    }

    # 获取Gaze-LLE模型预测
    with torch.no_grad():
        output = gaze_model(input)

    # 检测物体
    object_boxes = detect_objects(image, yolo_model)

    # 获取每个注视点
    gaze_points = []
    gazed_objects = []
    for i in range(len(bboxes)):
        if output['inout'][0][i] > 0.5:  # 只处理在画面内的注视
            gaze_point = get_gaze_target(output['heatmap'][0][i], (width, height))
            gazed_object = find_gazed_object(gaze_point, object_boxes, (width, height))
            gaze_points.append(gaze_point)
            gazed_objects.append(gazed_object)

    # 可视化结果
    result_image = visualize_results(image, gaze_points, object_boxes, gazed_objects)
    
    # 保存结果
    result_image.save(os.path.join(save_dir, 'gaze_object_detection_result.png'))
    print(f"Results saved to {os.path.join(save_dir, 'gaze_object_detection_result.png')}")

if __name__ == '__main__':
    main() 