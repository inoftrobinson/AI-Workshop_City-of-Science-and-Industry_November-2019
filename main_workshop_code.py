# Pour utiliser des modules, il faut les importer

# Ont importe notre modèle d'intelligence artificielle, et ses utilitaires
import time

from models import *
from utils import utils

# Importe une librairie pour générer des chiffres et valeurs aléatoires
import random

# Ont importe la librairie d'intelligence artificielle
import torch
from torchvision import transforms

# Ont importe ce qu'il faut pour afficher les résultats de l'intelligence artificielle
import matplotlib.pyplot as plt  # Librairie graphique
import matplotlib.patches as patches
from PIL import Image  # Pour gérer une image
import cv2  # Pour afficher et traiter de la vidéo (de la webcam par exemple)

# Ont definis où sous les fichiers du réseau de neurones entrainer
ai_configuration_chemin_daccess = 'config/yolov3.cfg'
ai_poids_chemin_daccess = 'config/yolov3.weights'
nom_des_objets_chemin_daccess = 'config/coco_fr.names'
image_size = 416  # Ont définis la résolution de l'image qu'ont vas donner à notre intelligence
conf_thres = 0.8
nms_thres = 0.4

# Ont charge notre model d'intelligence artificielle
model = Darknet(ai_configuration_chemin_daccess, img_size=image_size)

# Ont charge les poids (ce que notre intelligence artificielle à appris)
model.load_weights(ai_poids_chemin_daccess)
model.eval()

# Ont charger les noms de ce que notre intelligence a appris à reconnaitre
classes = utils.load_classes(nom_des_objets_chemin_daccess)

def detect_image(image: Image):
    # Modifie la taille de l'image, et ses marges pour qu'ont puisse envoyer n'importe quel type d'image à notre intelligence
    ratio = min(image_size/image.size[0], image_size/image.size[1])
    imw = round(image.size[0] * ratio)
    imh = round(image.size[1] * ratio)
    img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2), 0), max(int((imw-imh)/2), 0),
                         max(int((imh-imw)/2), 0), max(int((imw-imh)/2), 0)),
                        (128, 128, 128)), transforms.ToTensor(),

         ])

    # Ont convertis notre image dans un format optimisé pour les ordinateurs (un Tensor)
    image_tensor = img_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = torch.FloatTensor(image_tensor)
    # Donne notre image à notre intelligence, et récupère ce que notre intelligence à vus, sous la forme de prédictions
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]



cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]


video_depuis_la_webcam = cv2.VideoCapture(1)


# Crée la fenetre pour afficher la vidéo
fig, ax = plt.subplots(1, figsize=(16, 12), constrained_layout=False)
plt.title("Atelier à la cité des sciences et de l'industrie\nAnimer par Inoft")
plt.axis("off")

index_image = 0
time_to_process_last_image = 0
sec = 0

boites = list()
textes = list()

while True:
    time_when_starting_processing = time.time()
    sec = sec + time_to_process_last_image
    sec = round(sec, 2)

    # Ont supprime les boites et texte d'identifications de l'image précèdentes
    for boite in boites:
        boite.remove()
    boites = list()
    for texte in textes:
        texte.remove()
    textes = list()

    return_code, crude_frame = video_depuis_la_webcam.read()
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
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            # Crée des boites et labels de détections, pour chaque prédiction que notre machine a faites
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # Des maths pour définir la taille et position de nos boîtes
                box_h = ((y2 - y1) / unpad_h) * image.shape[0]
                box_w = ((x2 - x1) / unpad_w) * image.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * image.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * image.shape[1]
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

                # Dessine nos boites et textes, en les ajoutant sur un graphique vide
                boite = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor='none')
                boites.append(boite)
                ax.add_patch(boite)

                # Quand ont crée le texte cela peut crée des erreurs de temps en temps, donc ont lui dis que si il ne trouve pas correctement son texte, de ne pas planter
                try:
                    textes.append(plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top', bbox={'color': color, 'pad': 0}))
                except Exception:

                    pass

        # Puis montre l'image de la webcam par dessus les boites qu'ont viens de dessiner, pour qu'ont puisse voir ce qu'identifie notre machine
        ax.imshow(image)
        plt.pause(0.2)

        time_to_process_last_image = time.time() - time_when_starting_processing
        index_image += 1
