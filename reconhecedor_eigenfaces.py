import cv2

detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
#reconhecedor.read("classificadorEigen.yml")
reconhecedor.read("Recogniser/trainingDataEigen.xml")
largura, altura = 640, 480
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza)
    for (x, y, l, a) in facesDetectadas:
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,0,255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = ""
        if id == 1:
            nome = 'Grupo 1'
        elif id == 2:
            nome = 'Grupo 2'
        elif id == 3:
            nome = 'Grupo 3'
        elif id == 4:
            nome = 'Grupo 4'
        elif id == 5:
            nome = 'Grupo 5'
        elif id == 6:
            nome = 'Grupo 6'
        elif id == 7:
            nome = 'Grupo 7'
        elif id == 8:
            nome = 'Grupo 8'
        cv2.putText(imagem, nome, (x,y +(a+30)), font, 2, (255,255,255))
        cv2.putText(imagem, str(confianca), (x,y + (a+50)), font, 1, (255,255,255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()