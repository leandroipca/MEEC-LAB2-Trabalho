import cv2
import os
import numpy as np
from PIL import Image

eigenface = cv2.face.EigenFaceRecognizer_create(15)
fisherface = cv2.face.FisherFaceRecognizer_create(2)
lbph = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 15)

def getImagemComId():
    caminhos = [os.path.join('dataSet', f) for f in os.listdir('dataSet')]
   # print(caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:

       # >>>  Converte a imagem para tons cinzentos (modos 'L' - Library PIL) - Escolha do modo.

        imagemFace = Image.open(caminhoImagem).convert('L')
       # imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)

        imagemFace = imagemFace.resize((220, 220))
        imagemFace = np.array(imagemFace, 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split('_')[0].replace("G", ""))
        print(id)
        ids.append(id)
        faces.append(imagemFace)
        #cv2.imshow("Face", imagemFace)
        cv2.waitKey(10)
    return np.array(ids), faces

ids, faces = getImagemComId()
print(ids)
print(faces)

print("Treinando....")
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento realizado...")
