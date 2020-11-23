import os

# Pour utiliser des modules, il faut les importer

# Ont importe notre modèle d'intelligence artificielle, et ses utilitaires
from models import *
from utils import utils

# Ont importe la librairie d'intelligence artificielle
import torch
from torchvision import transforms
from torch.autograd import Variable

# Ont importe ce qu'il faut pour afficher les résultats de l'intelligence artificielle
import matplotlib.pyplot as plt  # Librairie graphique
from PIL import Image  # Pour gérer une image
import cv2  # Pour afficher et traiter de la vidéo (de la webcam par exemple)

# Ont definis où sous les fichiers du réseau de neurones entrainer
config_path = '../config/yolov3.cfg'
weights_path = '../config/yolov3.weights'
class_path = '../config/coco_fr.names'
image_size = 416
conf_thres = 0.8
nms_thres = 0.4

# Load model and weights
model = Darknet(config_path, img_size=image_size)
model.load_weights(weights_path)
# model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.FloatTensor  #torch.cuda.FloatTensor

def detect_image(image: Image):
    # scale and pad image
    ratio = min(image_size/image.size[0], image_size/image.size[1])
    imw = round(image.size[0] * ratio)
    imh = round(image.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]



cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# Charge la vidéo et met ses images dans l'ordre
from sort import *
videopath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos/jean_claude.mp4")
videopath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "videos/isa.wmv")
video = cv2.VideoCapture(0)
mot_tracker = Sort()


i_frame = 0
plt.ion()
plt.title("Inoft")
plt.axis("off")
# fig = plt.figure(figsize=(12, 8))

#success = video.set(cv2.CAP_PROP_FPS, 1)
#print(success)

time_to_process_last_image = 0
sec = 0

while True:
# for ii in range(40):
    time_when_starting_processing = time.time()
    sec = sec + time_to_process_last_image
    sec = round(sec, 2)
    # video.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)

    return_code, crude_frame = video.read()
    if crude_frame is not None:
        frame = cv2.cvtColor(crude_frame, cv2.COLOR_BGR2RGB)
        pilimg = Image.fromarray(frame)
        detections = detect_image(pilimg)

        image = np.array(pilimg)
        pad_x = max(image.shape[0] - image.shape[1], 0) * (image_size / max(image.shape))
        pad_y = max(image.shape[1] - image.shape[0], 0) * (image_size / max(image.shape))
        unpad_h = image_size - pad_y
        unpad_w = image_size - pad_x
        if detections is not None:
            tracked_objects = mot_tracker.update(detections.cpu())

            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
                box_h = int(((y2 - y1) / unpad_h) * image.shape[0])
                box_w = int(((x2 - x1) / unpad_w) * image.shape[1])
                y1 = int(((y1 - pad_y // 2) / unpad_h) * image.shape[0])
                x1 = int(((x1 - pad_x // 2) / unpad_w) * image.shape[1])

                color = colors[int(obj_id) % len(colors)]
                color = [i * 255 for i in color]
                cls = classes[int(cls_pred)]
                cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
                cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
                cv2.putText(frame, cls, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

            plt.rcParams["toolbar"] = "None"
            # new_crude_frame = cv2.resize(crude_frame, (int(1000), int(1000)))
            plt.imshow(frame)
            plt.pause(0.2)

        time_to_process_last_image = time.time() - time_when_starting_processing
        i_frame += 1
    else:
        # video.set(cv2.CAP_PROP_POS_MSEC, 0)
        print("reset")
        time_to_process_last_image = 0
        i_frame = 0
        sec = 0


