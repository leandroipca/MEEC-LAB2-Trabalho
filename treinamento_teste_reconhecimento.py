import cv2
import os
import PIL
import numpy as np
from PIL import Image

eigenface = cv2.face.EigenFaceRecognizer_create(15)
fisherface = cv2.face.FisherFaceRecognizer_create(2)
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('testes_reconhecimento/treinamento', f) for f in os.listdir('testes_reconhecimento/treinamento')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
       imagemFace = Image.open(caminhoImagem).convert('L')

       # Imagem com flip horizontal
       horizontal_imagem = imagemFace.transpose(PIL.Image.FLIP_LEFT_RIGHT)

       # Com a imagem invertida
       imagemNP = np.array(horizontal_imagem, 'uint8')

       # Com a imagem sem inversão
       #imagemNP = np.array(imagemFace, 'uint8')


       id = int(os.path.split(caminhoImagem)[1].split('_')[0].replace("G", ""))
       ids.append(id)
       faces.append(imagemNP)
    return np.array(ids), faces

ids, faces = getImagemComId()
print(ids)
print(faces)

print("Treinando...")

#eigenface.train(faces, ids)
#eigenface.write('classificadorEigen_Teste.yml')

#fisherface.train(faces, ids)
#fisherface.write('classificadorFisher_Teste.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH_Teste.yml')

print("Treinamento realizado")