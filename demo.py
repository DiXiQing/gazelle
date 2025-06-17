import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from retinaface import RetinaFace

def load_model():
    """Load Gaze-LLE model"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
    model.eval()
    model.to(device)
    return model, transform, device

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

def visualize_heatmap(pil_image, heatmap, bbox=None, inout_score=None):
    """Visualize gaze heatmap for a single person"""
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.Resampling.BILINEAR)
    heatmap = plt.cm.jet(np.array(heatmap) / 255.)
    heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
    heatmap = Image.fromarray(heatmap).convert("RGBA")
    heatmap.putalpha(90)
    overlay_image = Image.alpha_composite(pil_image.convert("RGBA"), heatmap)

    if bbox is not None:
        width, height = pil_image.size
        xmin, ymin, xmax, ymax = bbox
        draw = ImageDraw.Draw(overlay_image)
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], 
                      outline="lime", width=int(min(width, height) * 0.01))

        if inout_score is not None:
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill="lime", 
                     font=ImageFont.load_default(size=int(min(width, height) * 0.05)))
    return overlay_image

def visualize_all(pil_image, heatmaps, bboxes, inout_scores, inout_thresh=0.5):
    """Visualize all detected faces and their gaze directions"""
    colors = ['lime', 'tomato', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        draw.rectangle([xmin * width, ymin * height, xmax * width, ymax * height], 
                      outline=color, width=int(min(width, height) * 0.01))

        if inout_scores is not None:
            inout_score = inout_scores[i]
            text = f"in-frame: {inout_score:.2f}"
            text_width = draw.textlength(text)
            text_height = int(height * 0.01)
            text_x = xmin * width
            text_y = ymax * height + text_height
            draw.text((text_x, text_y), text, fill=color, 
                     font=ImageFont.load_default(size=int(min(width, height) * 0.05)))

        if inout_scores is not None and inout_score > inout_thresh:
            heatmap = heatmaps[i]
            heatmap_np = heatmap.detach().cpu().numpy()
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            draw.ellipse([(gaze_target_x-5, gaze_target_y-5), (gaze_target_x+5, gaze_target_y+5)], 
                        fill=color, width=int(0.005*min(width, height)))
            draw.line([(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)], 
                     fill=color, width=int(0.005*min(width, height)))

    return overlay_image

def main():
    # 设置图片路径和输出目录
    image_path = "testimg_gaze_4.png"  # 在这里修改你的图片路径
    save_dir = "output"
    
    # 创建输出目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型和图片
    model, transform, device = load_model()
    image = load_image(image_path)
    # 转换为RGB格式
    image = image.convert('RGB')
    width, height = image.size

    # 检测人脸
    bboxes = detect_faces(image)
    if not bboxes:
        print("No faces detected in the image!")
        return

    # 准备模型输入
    img_tensor = transform(image).unsqueeze(0).to(device)
    norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]

    input = {
        "images": img_tensor,
        "bboxes": norm_bboxes
    }

    # 获取模型预测
    with torch.no_grad():
        output = model(input)

    # 可视化每个人的热力图
    for i in range(len(bboxes)):
        plt.figure()
        result = visualize_heatmap(image, output['heatmap'][0][i], 
                                 norm_bboxes[0][i], 
                                 output['inout'][0][i] if output['inout'] is not None else None)
        plt.imshow(result)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'heatmap_{i}.png'))
        plt.close()

    # 可视化组合结果
    plt.figure(figsize=(10,10))
    result = visualize_all(image, output['heatmap'][0], norm_bboxes[0], 
                          output['inout'][0] if output['inout'] is not None else None)
    plt.imshow(result)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'combined_result.png'))
    plt.close()

if __name__ == '__main__':
    main()