"""
########################## Explicação sobre Acurácia, Sensibilidade e especificidade #################################

Acurácia
É a proporção de predições corretas, sem considerar o que é positivo e o que negativo e sim o acerto total. É dada por:

ACC=(VP+VN)/(P+N)

em que p é o número de eventos (Y=1, chamado aqui de positivo) e n é o número de não eventos
(^Y=0, chamado aqui de negativo).

Sensibilidade
É a proporção de verdadeiros positivos, ou seja, avalia a capacidade do modelo classificar um indivíduo como evento
$ (^Y=1) $ dado que realmente ele é evento (Y=1):

SENS=VP/(VP+FN)

Especificidade

É a proporção de verdadeiros negativos, isto é, avalia a capacidade do modelo predizer um indivíduo como não evento
$ (\hat{Y}=0) $ dado que ele realmente é não evento (Y=0).

ESPEC=VN/(VN+FP)
######################################################################################################################
"""

"""
######################################## Bloco de treimanento #######################################################
# bloco de treinamento 
import cv2
import os
import PIL
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

eigenface = cv2.face.EigenFaceRecognizer_create(15)
fisherface = cv2.face.FisherFaceRecognizer_create(2)
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    caminhos = [os.path.join('testes_reconhecimento/crop', f) for f in os.listdir('testes_reconhecimento/crop')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
       imagemFace = Image.open(caminhoImagem).convert('L')

       # Imagem com flip horizontal
       #horizontal_imagem = imagemFace.transpose(PIL.Image.FLIP_LEFT_RIGHT)

       # Com a imagem invertida
       #imagemNP = np.array(horizontal_imagem, 'uint8')

       # Com a imagem sem inversão
       imagemNP = np.array(imagemFace, 'uint8')


       id = int(os.path.split(caminhoImagem)[1].split('_')[0].replace("G", ""))
       ids.append(id)
       faces.append(imagemNP)
    return np.array(ids), faces

ids, faces = getImagemComId()

print("Treinando...")

#eigenface.train(faces, ids)
#eigenface.write('classificadorEigen_Teste.yml')

#fisherface.train(faces, ids)
#fisherface.write('classificadorFisher_Teste.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH_Teste.yml')

print("Treinamento realizado")


# fim do bloco de treinamento
#######################################################################################################################
"""


from typing import List
import PIL
from PIL import Image
import sys

# Image I/O
import imageio

import cv2
import os

import numpy as np
from skimage import metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support, accuracy_score, \
    classification_report

detectorFace = cv2.CascadeClassifier("Haar/haarcascade_frontalface_alt.xml")

"""
# Função aumentar brilho
def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def increase_contrast(img):

    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    #cv2.imshow('Aumento de contraste', img2)
    return img2
"""

# >>>> Tipos de classificador

#reconhecedorEingen = cv2.face.EigenFaceRecognizer_create()
#reconhecedorEingen.read("classificadorEigen_Teste.yml")
#reconhecedorFisher = cv2.face.FisherFaceRecognizer_create()
#reconhecedorFisher.read("classificadorFisher_Teste.yml")
#reconhecedorLBPH = cv2.face.LBPHFaceRecognizer_create()
reconhecedorLBPH = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 15)
reconhecedorLBPH.read("classificadorLBPH_Teste.yml")

totalAcertosLBPH = 0
percentualAcertoLBPH = 0.0
totalConfiancaLBPH = 0.0
cont = 0

Y_Predict = []
Y_test = []
Y_Train =[]
X_Train = []
X_Test = []

caminhos = [os.path.join('testes_reconhecimento/teste', f) for f in os.listdir('testes_reconhecimento/teste')]

# Classificador LBPH
for caminhoImagem in caminhos:
    # Ler a imagem
    imagem = cv2.imread(caminhoImagem)

    # Converte a cinza pelo PIL
    imagemFace = Image.open(caminhoImagem).convert('L')

    #Imagens em NP
    imagemFaceNP = np.array(imagemFace, 'uint8')

    # Detecção de imagems
    facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP, minNeighbors=8)

    for (x, y, l, a) in facesDetectadas:

        # Detecção facial
        cv2.rectangle(imagemFaceNP, (x, y), (x + l, y + a), (255, 0, 255), 2)

        # Dados preditos
        idprevistoLBPH, confiancaLBPH = reconhecedorLBPH.predict(imagemFaceNP)

        # Split das imagens
        idatual = int(os.path.split(caminhoImagem)[1].split('_')[0].replace("G", ""))

        # Resultados
        print(str(idatual) + " foi classificado como " + str(idprevistoLBPH) + " - " + "Confiança: " +
              str(totalConfiancaLBPH))
        cont += 1


        # Adicionando na lista os valores previstos pelo reconhecedor
        Y_Predict.append(idprevistoLBPH)

        # Adicionando na lista os valores verdadeiros
        Y_test.append(idatual)

        print("y_previsto", Y_Predict)
        print("y_verdadeiro", Y_test)


        if idprevistoLBPH == idatual:
            totalAcertosLBPH += 1
            totalConfiancaLBPH += confiancaLBPH
        else:
            totalAcertosLBPH -= 1

        print("Total de Acertos", totalAcertosLBPH)
        #cv2.imshow("LBPH sem tratamento", imagemFaceNP)
        cv2.waitKey(1000)



conf = confusion_matrix(Y_test, Y_Predict)
print(conf)

prfs = precision_recall_fscore_support(Y_test, Y_Predict)
print("Precision Recall F-score Support:")
print(prfs)

accuracy = accuracy_score(Y_test, Y_Predict)
print("Accuracy:")
print(accuracy)

cr=classification_report(Y_test, Y_Predict)
print("Classification Report:")
print(cr)


"""
print ('Total de imagens testadas = ', cont)
print('-=' * 30)


percentualAcertoLBPH = (totalAcertosLBPH / cont) * 100
#totalConfiancaLBPH = totalConfiancaLBPH
print('Percentual de acerto: ' + str(percentualAcertoLBPH) + '%')
#print("Total confiança: " + str(totalConfiancaLBPH))
"""