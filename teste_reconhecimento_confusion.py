from typing import List
import PIL
from PIL import Image
import sys

# Image I/O
import imageio

import cv2
import os

import numpy as np

from openpyxl import Workbook

arquivo_excel = Workbook()
from openpyxl import load_workbook
arquivo_excel = load_workbook('resultados.xlsx')


detectorFace = cv2.CascadeClassifier("Haar/haarcascade_frontalface_alt.xml")

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


# >>>> Tipos de classificador

# reconhecedorEingen = cv2.face.EigenFaceRecognizer_create()
# reconhecedorEingen.read("classificadorEigen_Teste.yml")
# reconhecedorFisher = cv2.face.FisherFaceRecognizer_create()
# reconhecedorFisher.read("classificadorFisher_Teste.yml")
#reconhecedorLBPH = cv2.face.LBPHFaceRecognizer_create()
reconhecedorLBPH = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 15)
reconhecedorLBPH.read("classificadorLBPH_Teste.yml")

totalAcertosLBPH = 0
percentualAcertoLBPH = 0.0
totalConfiancaLBPH = totalConfiancaLBPH_Brilho = totalConfiancaLBPH_Escuro = totalConfiancaLBPH_Blur = 0.0
cont = 0



# Dados Matriz Confusão
matriz = [[0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0]]

FP = FP_Brilho = FP_Escuro=  FP_Blur = 0
FN = FN_Brilho = FN_Escuro = FN_Blur = 0
TPTF = 0

caminhos = [os.path.join('testes_reconhecimento/teste', f) for f in os.listdir('testes_reconhecimento/teste')]

# Classificador LBPH
for caminhoImagem in caminhos:
    # Ler a imagem
    #imagem = imageio.imread(caminhoImagem)
    imagem = cv2.imread(caminhoImagem)

    # Ajusta o brilho da imagem
    imagem_brilho = increase_brightness(imagem, 50)

    # Ajusta o contraste da imagem
    imagemEscura = increase_contrast(imagem)

    # Efeito Blur na imagem
    blurImg = cv2.blur(imagem, (20, 20))

    # Converte a cinza pelo PIL
    imagemFace = Image.open(caminhoImagem).convert('L')

    # Converte a imagem com brilho para cinza pelo cv2
    imagemCinza_Brilho = cv2.cvtColor(imagem_brilho, cv2.COLOR_BGR2GRAY)

    # Converte a imagem com mais contraste para cinza pelo cv2
    imagemCinza_Escura = cv2.cvtColor(imagemEscura, cv2.COLOR_BGR2GRAY)

    # Converte a imagem com blur para cinza pelo cv2
    blurImgCinza = cv2.cvtColor(blurImg, cv2.COLOR_BGR2GRAY)

    # Imagem com flip horizontal
    #horizontal_imagem = imagemFace.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    # Imagem rotacionada em 10 Graus
    #rotate_imagem = imagemFace.rotate(10)

    #Imagens em NP
    imagemFaceNP = np.array(imagemFace, 'uint8')
    brilho_imagemNP = np.array(imagemCinza_Brilho, 'uint8')
    escuro_imagemNP = np.array(imagemCinza_Escura, 'uint8')
    blurImgNP = np.array(blurImgCinza, 'uint8')
    # horizontalFaceNP = np.array(horizontal_imagem, 'uint8')
    # rotate_imagemNP = np.array(rotate_imagem, 'uint8')

    # Detecção de imagems
    facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP, minNeighbors=8)
    facesDetectadas_brilho = detectorFace.detectMultiScale(brilho_imagemNP, minNeighbors=8)
    facesDetectadas_escura = detectorFace.detectMultiScale(escuro_imagemNP, minNeighbors=8)
    facesDetectadas_blur = detectorFace.detectMultiScale(blurImgNP, minNeighbors=8)
    #facesDetectadas_flip = detectorFace.detectMultiScale(horizontalFaceNP, minNeighbors=8)
    #facesDetectadas_rotate = detectorFace.detectMultiScale(rotate_imagemNP, minNeighbors=8)

    # For das faces sem tratamento
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
        if idprevistoLBPH == idatual:
        #if idprevistoLBPH == 8:
            totalAcertosLBPH += 1
            #print (idatual)
            #print (idprevistoLBPH)
            matriz[idatual-1][idprevistoLBPH-1] += 1
            totalConfiancaLBPH += confiancaLBPH
            TPTF += 1
        else:
            totalAcertosLBPH -= 1
            matriz[idatual - 1][idprevistoLBPH - 1] += 1
            FN += 1
            FP += 1

        print("Total de Acertos", totalAcertosLBPH)
        cv2.imshow("LBPH sem tratamento", imagemFaceNP)
        cv2.waitKey(1000)

    # For das imagens com brilho
    for (x, y, l, a) in facesDetectadas_brilho:
        # Detecção facial
        cv2.rectangle(brilho_imagemNP, (x, y), (x + l, y + a), (255, 0, 255), 2)

        # Dados preditos
        idprevistoLBPH_Brilho, confiancaLBPH_Brilho = reconhecedorLBPH.predict(brilho_imagemNP)

        # Split das imagens
        idatual = int(os.path.split(caminhoImagem)[1].split('_')[0].replace("G", ""))

        # Resultados
        print(str(idatual) + " foi classificado em modo Brilho como " + str(idprevistoLBPH_Brilho) + " - " + "Confiança: " +
               str(totalConfiancaLBPH_Brilho))
        cont += 1
        if idprevistoLBPH_Brilho == idatual:
        #if idprevistoLBPH_Brilho == 8:
            totalAcertosLBPH += 1
            matriz[idatual - 1][idprevistoLBPH_Brilho - 1] += 1
            totalConfiancaLBPH_Brilho += confiancaLBPH_Brilho
            TPTF += 1

        else:
            totalAcertosLBPH -= 1
            matriz[idatual - 1][idprevistoLBPH_Brilho - 1] += 1
            FN_Brilho += 1
            FP_Brilho += 1

        print("Total de Acertos", totalAcertosLBPH)
        cv2.imshow("Brilho", brilho_imagemNP)
        cv2.waitKey(1000)

    # For das imagens escurecidas
    for (x, y, l, a) in facesDetectadas_escura:
        # Detecção facial
        cv2.rectangle(escuro_imagemNP, (x, y), (x + l, y + a), (255, 0, 255), 2)

        # Dados preditos
        idprevistoLBPH_Escuro, confiancaLBPH_Escuro = reconhecedorLBPH.predict(escuro_imagemNP)

        # Split das imagens
        idatual = int(os.path.split(caminhoImagem)[1].split('_')[0].replace("G", ""))

        # Resultados
        print(str(idatual) + " foi classificado em modo Escuro como " + str(idprevistoLBPH_Escuro) + " - " + "Confiança: " +
               str(totalConfiancaLBPH_Escuro))
        cont += 1
        if idprevistoLBPH_Escuro == idatual:
        #if idprevistoLBPH_Escuro == 8:
            totalAcertosLBPH += 1
            matriz[idatual - 1][idprevistoLBPH_Escuro - 1] += 1
            totalConfiancaLBPH_Escuro += confiancaLBPH_Escuro
            TPTF += 1
        else:
            totalAcertosLBPH -= 1
            matriz[idatual - 1][idprevistoLBPH_Escuro - 1] += 1
            FN_Escuro += 1
            FP_Escuro += 1

        print("Total de Acertos", totalAcertosLBPH)
        cv2.imshow("Escuro", escuro_imagemNP)
        cv2.waitKey(1000)

    # For das imagens com blur
    for (x, y, l, a) in facesDetectadas_blur:
        # Detecção facial
        cv2.rectangle(blurImgNP, (x, y), (x + l, y + a), (255, 0, 255), 2)

        # Dados preditos
        idprevistoLBPH_Blur, confiancaLBPH_Blur = reconhecedorLBPH.predict(blurImgNP)

        # Split das imagens
        idatual = int(os.path.split(caminhoImagem)[1].split('_')[0].replace("G", ""))

        # Resultados
        print(str(idatual) + " foi classificado efeito Blur como " + str(idprevistoLBPH_Blur) + " - " + "Confiança: " +
               str(totalConfiancaLBPH_Blur))
        cont += 1
        if idprevistoLBPH_Blur == idatual:
        #if idprevistoLBPH_Blur == 8:
            totalAcertosLBPH += 1
            matriz[idatual - 1][idprevistoLBPH_Blur - 1] += 1
            totalConfiancaLBPH_Blur += confiancaLBPH_Blur
            TPTF += 1
        else:
            totalAcertosLBPH -= 1
            matriz[idatual - 1][idprevistoLBPH_Blur - 1] += 1
            FN_Blur += 1
            FP_Blur += 1

        print("Total de Acertos", totalAcertosLBPH)
        cv2.imshow("Efeito Blur", blurImgNP)
        cv2.waitKey(1000)

FN = FN + FN_Brilho + FN_Escuro + FN_Blur
FP = FP + FP_Brilho + FP_Escuro + FP_Blur

accuracy = TPTF/cont
precision = TPTF/(TPTF+FP)
recall = TPTF/(TPTF+FN)
F1 = ((2 * precision * recall)/(precision + recall))


print ('Matriz de Resultados: ')
print (np.matrix(matriz))
print ('Falso Positivo = ', FP)
print ('Falso Negativo = ', FN)
print ('Total de TP e TF = ', TPTF)
print ('Total de imagens testadas', cont)
print ('============================================')
print ('Accuracy = ', accuracy)
print ('Precision = ', precision)
print ('Recall = ', recall)
print ('F1 Score = ', F1)
print ('============================================')


percentualAcertoLBPH = (totalAcertosLBPH / cont) * 100
#totalConfiancaLBPH = totalConfiancaLBPH
print('Percentual de acerto: ' + str(percentualAcertoLBPH) + '%')
#print("Total confiança: " + str(totalConfiancaLBPH))
