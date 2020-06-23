import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

lbph = cv2.face.LBPHFaceRecognizer_create()


def getImagemComId():
    caminhos = [os.path.join('testes_reconhecimento/treinamento_todos', f) for f in
                os.listdir('testes_reconhecimento/treinamento_todos')]
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = Image.open(caminhoImagem).convert('L')

        # Imagem com flip horizontal
        # horizontal_imagem = imagemFace.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # Com a imagem invertida
        # imagemNP = np.array(horizontal_imagem, 'uint8')

        # Com a imagem sem inversão
        imagemNP = np.array(imagemFace, 'uint8')

        id = int(os.path.split(caminhoImagem)[1].split('_')[0].replace("G", ""))
        ids.append(id)
        faces.append(imagemNP)
    return np.array(ids), faces


ids, faces = getImagemComId()
# print("Y", ids)
# print("X", faces)

# print ("tamanho ids ", len.ids )
# print ("tamanho faces", len.faces )

print("Treinando...")

Y = ids
X = np.array(faces)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

lbph.train(faces, ids)
#lbph.train(X_train, Y_train)
lbph.write('classificadorLBPH_Teste.yml')

print("Treinamento realizado")

"""
nsamples, nx, ny = X_train.shape
X_train_2d = X_train.reshape((nsamples, nx * ny))
print("Tamanho X_train 2d", X_train_2d.shape)

nsamples, nx, ny = X_test.shape
X_test_2d = X_test.reshape((nsamples, nx * ny))
print("Tamanho X_test 2d", X_test_2d.shape)
"""

# Y_predict = lbph.predict(X_test)
Y_predict = np.zeros((len(Y_test),),dtype=int)
for i in range(len(X_test)):
    a = lbph.predict(X_test[i,:,:])
    Y_predict[i] = a[0]
    #Y_predict[i] = lbph.predict(X_test[i,:,:])

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
