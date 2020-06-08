#######################################################################################################################
# ------------------------------------------------- Breve detalhamento ------------------------------------------------#
#######################################################################################################################
# Confiança corresponde à medida entre a imagem facial submetida e a imagem facial classificada.
# Quanto mais próximo de 1 melhor será.
#
# >>> Parametros LBPH (radius, neighbors, grid_x, grid_y, threshold):
# >> radius: Raio maior aumenta a abrangência mas pode perder bordas finas (pontos mais distantes).
#   Quanto maior o raio mais padrões podem ser codificados, mas aumenta o esforço computacional.
# >> neighbors:  Número de pontos da amostra para construir um padrão local. Quanto maior o número de vizinhos maior é
#   o esforço computacional
# >> grid_x: Número de células na horizontal. Quanto mais células maior é a dimensionalidade do vetor de
#   características (histogramas)
# >> grid_y: Número de células na vertical. Se a grade aumentar serão usados menos pixels em cada histograma
#   (mais esparços)
# >> threshold: Limite de confiança
#
# --------------------------------------------Script do Grupo 8-------------------------------------------------------#
#######################################################################################################################

# -*- coding: utf-8 -*-
import cv2

# >>> Cores em BGR
branco = [255, 255, 255]
azul_navy = [139, 0, 0]
vermelho_escuro = [0, 0, 139]

# >>> Escolher qual metodo de detecção de faces
detectorFace = cv2.CascadeClassifier("Haar/haarcascade_frontalface_default.xml")
# detectorFace = cv2.CascadeClassifier("Haar/haarcascade_frontalface_alt.xml")
# detectorFace = cv2.CascadeClassifier("Haar/haarcascade_frontalcatface.xml")

# >>> A utilizar o LBPH com as características
reconhecedor = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 15)

# >>> Escolher a base (treinamento com redimensionamento ou sem)
# reconhecedor.read("Recogniser/trainingDataLBPH.xml") # classificador de teste
reconhecedor.read("classificadorLBPH.yml")  # classificador sem redimensionamento

# >>> Escolher a escala de redimensionamento
largura, altura = 220, 220
# largura, altura = 640 , 480
# largura, altura = 1280 , 720

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza)
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), vermelho_escuro, 2)

        # >>> Escolher se vai utilizar imagem redimensionada ou não
        id, confianca = reconhecedor.predict(imagemFace)  # usando a imagem redimensionada
        # id, confianca = reconhecedor.predict(imagemCinza) #usando a imagem sem redimensionamento

        nome = ""
        if id == 8:
            nome = 'Grupo 8'
        else:
            nome = 'Nao Identificado'
        cv2.putText(imagem, nome, (x, y + (a + 30)), font, 2, branco)
        cv2.putText(imagem, 'Confianca: ', (x, y + (a + 50)), font, 1, branco)
        cv2.putText(imagem, str(round(confianca)), (x + 150, y + (a + 50)), font, 1, branco)

    cv2.imshow("Reconhecimento por LBPH", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
