import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box


def create_model(num_classes):
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device="cpu"
print("using {} device.".format(device))

# create model
model = create_model(num_classes=21)

# load train weights
train_weights = "./save_weights/model.pth"
assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
model.to(device)

# read class_indict
label_json_path = './pascal_voc_classes.json'
assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
json_file = open(label_json_path, 'r')
class_dict = json.load(json_file)
category_index = {v: k for k, v in class_dict.items()}


def main(image_path):
    # load image
    # "./test.jpg"
    original_img = Image.open(image_path) 

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
   
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        draw_box(original_img,
                 predict_boxes,
                 predict_classes,
                 predict_scores,
                 category_index,
                 thresh=0.5,
                 line_thickness=3)
        # plt.imshow(original_img)
        # plt.show()
        # 保存预测的图片结果
        image_name="./IMG/" + image_path.split("/")[-1]
        original_img.save(image_name)



# 将帧组合成视频
def frame2video(image_path):
    import cv2
    image_list=os.listdir(image_path)
    image_list.sort()
    # 第一张图片
    first_image = cv2.imread( image_path + '/' + image_list[0])
    fps = 10
    print('fps: ',fps)
    # size
    size= (first_image.shape[1],first_image.shape[0])
    print(size)
    # 编码器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # MJPG
    # videowriter
    videoWrite = cv2.VideoWriter('./out.mp4',fourcc,fps,size)
    for image in image_list:
        print(image)
        image=image_path + '/' + image
        img = cv2.imread(image,cv2.IMREAD_COLOR)
        # 调整大小 
        img = cv2.resize(img,size,interpolation=cv2.INTER_CUBIC) 
        # 写
        videoWrite.write(img)
    print('video write success')

if __name__ == '__main__':
    
    root="./Demo"
    
    image_list=os.listdir(root)
    image_list.sort()
    for image in image_list:
        main(root + "/" + image)

    
    # frame2video("IMG")
