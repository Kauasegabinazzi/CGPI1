import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from get_images.coletor_dataset import baixar_e_extrair_dataset

# carregamento do modelo pré-treinado, créditos: https://github.com/Furkan-Gulsen/Face-Mask-Detection/blob/master/model.h5
model = load_model("files/model.h5")

# carregar detector de rostos do OpenCV
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

def detect_faces(image):
    (h, w) = image.shape[:2]
    # Pré-processamento da imagem para a rede DNN
    blob = cv2.dnn.blobFromImage(
        image, 
        scalefactor=1.0, 
        size=(300, 300), 
        mean=(104.0, 177.0, 123.0)  # Valores médios subtraídos (normalização)
    )
    net.setInput(blob)
    detections = net.forward()  # Executa a detecção
    print(detections)
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # Confiança da detecção
        if confidence > 0.3:  # Filtro de confiança (ajuste conforme necessário)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            faces.append((x, y, x2 - x, y2 - y))  # Converte para (x, y, w, h)
    
    return faces

# Função para detectar máscaras
def detect_mask(image):
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
    faces = detect_faces(image)
    print(faces)
    
    mask_count = 0
    no_mask_count = 0
    
    for (x, y, w, h) in faces:
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))  # Redimensionar para o modelo
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        
        prediction = model.predict(face)
        mask, no_mask = prediction[0]
        
        if mask > no_mask:
            mask_count += 1
            color = (0, 255, 0)  # Verde (com máscara)
        else:
            no_mask_count += 1
            color = (255, 0, 0)  # Vermelho (sem máscara)
        
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    
    return image, mask_count, no_mask_count

dataset_kaggle = "andrewmvd/face-mask-detection"
baixar_e_extrair_dataset(dataset_kaggle)

exp_len = 5

rnd_nmbrs = [str(p) for p in random.sample(range(853), exp_len)]
print(rnd_nmbrs)
img_path = "maksssksksss"
img_cont = [img_path + r for r in rnd_nmbrs]

for i in img_cont:
    # Carregar imagem
    image = cv2.imread(f"datasets/images/{i}.png")

    # Detectar máscaras
    result, mask, no_mask = detect_mask(image)

    print(f"Pessoas com máscara: {mask}")
    print(f"Pessoas sem máscara: {no_mask}")

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    # Mostrar imagem
    import matplotlib.pyplot as plt
    plt.imshow(result)
    plt.show()