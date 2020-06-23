'''
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

'''


from typing import List
import PIL
from PIL import Image
import sys

# Image I/O
import imageio

import cv2
import os

import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from openpyxl import Workbook

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
Y_predict = []
Y_test = []

caminhos = [os.path.join('testes_reconhecimento/hard_test', f) for f in os.listdir('testes_reconhecimento/hard_test')]


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
        Y_predict.append(idprevistoLBPH)
        Y_test.append(idatual)
        #print("Y_predict", Y_predict)
        #print("Y_test", Y_test)
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
        #cv2.imshow("LBPH sem tratamento", imagemFaceNP)
        cv2.waitKey(1000)
    """
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
        #cv2.imshow("Brilho", brilho_imagemNP)
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
        #cv2.imshow("Escuro", escuro_imagemNP)
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
        #cv2.imshow("Efeito Blur", blurImgNP)
        cv2.waitKey(1000)
    """

# outra abordagem

cm = metrics.confusion_matrix(Y_test, Y_predict, labels=[1, 2, 3, 4, 5, 6, 7, 8])
print("Confusion Matrix:")
print(cm)

prfs = metrics.precision_recall_fscore_support(Y_test, Y_predict)
print("Precision Recall F-score Support:")
print(prfs)

accuracy = metrics.accuracy_score(Y_test, Y_predict)
print("Accuracy:")
print(accuracy)

cr = metrics.classification_report(Y_test, Y_predict)
print("Classification Report:")
print(cr)


"""
# Calculos gerais sobre a matriz de confusão

FN = FN + FN_Brilho + FN_Escuro + FN_Blur
FP = FP + FP_Brilho + FP_Escuro + FP_Blur

accuracy = TPTF/cont
precision_All = TPTF/(TPTF+FP)
recall_All = TPTF/(TPTF+FN)
F1 = ((2 * precision_All * recall_All)/(precision_All + recall_All))

def precision(classe):
    #l = linha / c = coluna
    c  = classe - 1
    den = 0
    for l in range(0,8):
        den += matriz[l][c]
    precision = (((matriz[c][c])/den)*100)
    return precision

def recall(classe):
    #l = linha / c = coluna
    l  = classe - 1
    den = 0
    for c in range(0,8):
        den += matriz[l][c]
    recall = (((matriz[l][l])/den)*100)
    return recall

def specificity(classe):
    #l = linha / c = coluna
    l = classe - 1
    c = classe - 1
    soma = 0
    den = 0
    for c in range(0,8):
        soma += matriz[l][c]
    c = classe - 1
    for l in range(0,8):
        soma += matriz[l][c]
        den += matriz[l][c]
    TN = (cont-(soma-matriz[classe-1][classe-1]))
    specificity = (TN/(TN+(den-matriz[classe-1][classe-1])))
    return specificity

print('-=' * 30)
print('Precisão de cada foto')
print('-=' * 30)
print ('Precisão Foto 1 = ', precision(1))
print ('Precisão Foto 2 = ', precision(2))
print ('Precisão Foto 3 = ', precision(3))
print ('Precisão Foto 4 = ', precision(4))
print ('Precisão Foto 5 = ', precision(5))
print ('Precisão Foto 6 = ', precision(6))
print ('Precisão Foto 7 = ', precision(7))
print ('Precisão Foto 8 = ', precision(8))
print('-=' * 30)
print('')
print('-=' * 30)
print('Recall (Sensibilidade) de cada foto')
print('-=' * 30)
print ('Recall Foto 1 = ', recall(1))
print ('Recall Foto 2 = ', recall(2))
print ('Recall Foto 3 = ', recall(3))
print ('Recall Foto 4 = ', recall(4))
print ('Recall Foto 5 = ', recall(5))
print ('Recall Foto 6 = ', recall(6))
print ('Recall Foto 7 = ', recall(7))
print ('Recall Foto 8 = ', recall(8))
print('-=' * 30)
print('')
print('-=' * 30)
print('Specificity de cada foto')
print('-=' * 30)
print ('Specificity Foto 1 = ', specificity(1))
print ('Specificity Foto 2 = ', specificity(2))
print ('Specificity Foto 3 = ', specificity(3))
print ('Specificity Foto 4 = ', specificity(4))
print ('Specificity Foto 5 = ', specificity(5))
print ('Specificity Foto 6 = ', specificity(6))
print ('Specificity Foto 7 = ', specificity(7))
print ('Specificity Foto 8 = ', specificity(8))
print('-=' * 30)
print('')
print('-=' * 30)
print ('Matriz de Confusão: ')
print (np.matrix(matriz))
print('-=' * 30)
print('')
print ('Falso Positivo = ', FP)
print ('Falso Negativo = ', FN)
print ('Total de TPTN Todos = ', TPTF)
print ('Total de imagens testadas = ', cont)
print('-=' * 30)
print ('Accuracy Total= ', accuracy)
print ('Precision Total = ', precision_All)
print ('Recall Total = ', recall_All)
print ('F1 Score = ', F1)
print('-=' * 30)


percentualAcertoLBPH = (totalAcertosLBPH / cont) * 100
#totalConfiancaLBPH = totalConfiancaLBPH
print('Percentual de acerto: ' + str(percentualAcertoLBPH) + '%')
#print("Total confiança: " + str(totalConfiancaLBPH))
"""