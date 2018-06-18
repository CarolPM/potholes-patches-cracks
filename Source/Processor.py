import numpy as np
import cv2
import math
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import classification_report
import time
import warnings
global_features = []
global_labels = []
global_label_names = []
global_classifiers = []
global_classifiers_names = []

'''Funcao que transforma a imagem para escala em preto e branco, binariza, aplica erosao, dilatacao e, por fim, nova erosao
e em seguida a imagem binarizada eh invertida e retornada'''
def transform_image (img):
	thresh = 127
	maxValue = 255
	kernel = np.ones((5,5),np.uint8)
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, th = cv2.threshold(gray,thresh,maxValue,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
	erosion = cv2.erode(opening,kernel,iterations = 1)
	erosion = cv2.bitwise_not(erosion)
	return cv2.findContours(erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

'''Funcao que retorna a quantidade de contornos de uma imagem'''
def num_contours (contours):
	return len(contours)

'''Funcao que retorna a area de um contorno'''
def area (contour):
        a = cv2.contourArea(contour)
        if a == None:
                a = 0.0
        return a

'''Funcao que retorna o comprimento de um contorno'''
def length (contour):
        l = cv2.arcLength(contour,True)
        if l == None:
                l = 0.0
        return l

'''Funcao que retorna apenas os contornos com area dentro de um range'''
def contour_filter(area):
        a = cv2.contourArea(area)
        if a == None:
                a = 0.0
        return a >= 300 and a <= 2000

'''Funcao que retorna a largura de um contorno'''
def width (contour):
        x,y,w,h = cv2.boundingRect(contour)
        if w == None:
                w = 0.0
        return w

'''Funcao que retorna a altura de um contorno'''
def heigth (contour):
        x,y,w,h = cv2.boundingRect(contour)
        if h == None:
                h = 0.0
        return h

'''Funcao que retorna a circularidade de um contorno'''
def circularity (contour):
        c =(4*math.pi*cv2.contourArea(contour))/((cv2.arcLength(contour,True)**2))
        if c == None:
                c = 0.0
        return c

'''Funcao que retorna o alongamento de um contorno'''
def elongation (m):
        x = m['mu20'] + m['mu02']
        y = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
        e = (x + y**0.5) / (x - y**0.5)
        if e == None:
                e = 0.0
        return e

'''Funcao que retorna o desvio padrao de um contorno'''
def standard_deviation (imgcontour):
        (means, std) = cv2.meanStdDev(imgcontour)
        for k in std:
                for m in k:
                        std = m
        if std == None:
                std = 0.0
        return std

'''Funcao que salva as imagens processadas na pasta destino especificada'''
def save_image(img, destination_img):
        cv2.imwrite(destination_img, img)
        

def main():
        #Pasta origem dos contornos Pre-Processados
        source = 'Cropped_Resized_Data'
        #Pasta destino dos contornos retornados pelos classificadores "Nearest Neighbors", "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "Naive Bayes"
        '''destination = 'Contours/Predict_Proba_Classifiers' '''
        #Pasta destino dos contornos retornados pelos classificadores "Linear SVM", "RBF SVM", "AdaBoost", "QDA"
        destination = 'Contours/Decision_Function_Classifiers' 
        path_s = os.listdir(source)
        path_d = range(len(path_s))
        dictionary = {}
        #Para classificar entre buraco, rachadura, mancha e algo irrelevante
        dictionary['target_names'] = np.array(['pothole', 'crack', 'patch', 'irrelevant'], dtype='|S9')
        #Para classificar apenas entre buraco e algo irrelevante
        '''dictionary['target_names'] = np.array(['pothole', 'irrelevant'], dtype='|S9')'''
        dictionary['target'] = []
        dictionary['image_name'] = []
        dictionary['feature_names'] = np.array(['area', 'length', 'width', 'heigth', 'circularity', 'elongation', 'standard deviation'], dtype='|S23')
        dictionary['data'] = []
        list_name = []
        list_features = []
        
        #Predict Proba Algorithms Names
        '''classifiers_names = [
                "Nearest Neighbors",
                "Gaussian Process",
                "Decision Tree",
                "Random Forest",
                "Neural Net",
                "Naive Bayes"
                ]'''
        #Decision Function Algorithms Names
        classifiers_names = [
                "Linear SVM",
                "RBF SVM",
                "AdaBoost",
                "QDA"
                ]
        
        global global_classifiers_names
        global_classifiers_names = classifiers_names
        #Predict Proba Algorithms
        '''classifiers = [
            KNeighborsClassifier(3),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1),
            GaussianNB()
            ]'''
        #Decision Function Algorithms
        classifiers = [
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            AdaBoostClassifier(),
            QuadraticDiscriminantAnalysis()
            ]
        
        global global_classifiers
        global_classifiers = classifiers

        totalNumContours = 0;

        for i in range(0, len(path_s)):
                list_name.append(path_s[i])
                path_d[i] = os.path.join(destination, path_s[i])
                path_s[i] = os.path.join(source, path_s[i])
                if os.path.isfile(path_s[i]):
                        arquivos = path_s[i]
                else:
                        continue
                if arquivos.lower().endswith('.jpg'):
                        
                        jpgs = arquivos
                else:
                        continue
                img = cv2.imread(jpgs,1)
                
                imgcontours, contours, hierarchy = transform_image(img)
                #cv2.imshow('Contornos',imgcontours)
                contoursRange = contours
                contoursRange = filter(contour_filter, contours)
                
                j = 0
                imgcnt = np.zeros(imgcontours.shape[:2])
                imag = img.copy()
                imgsave = img.copy()
                print('\n------------------------------------------------------------')
                print('Image {} ({})' .format(i+1, list_name[i]))
                print ('Number of Contours: {}\n'.format(num_contours(contoursRange)))
                totalNumContours += num_contours(contoursRange)
                for contour in contoursRange:
                        lista = []
                        a = area(contour)
                        print ('Area[{}]: {}' .format(j, a))
                        l = length(contour)
                        print ('Length[{}]: {}' .format(j, l))
                        w = width(contour)
                        print ('Width[{}]: {}' .format(j, w))
                        h = heigth(contour)
                        print ('Heigth[{}]: {}' .format(j, h))
                        c = circularity(contour)
                        print ('Circularity[{}]: {}' .format(j, c))
                        m = cv2.moments(contour)
                        e = elongation(m)
                        print ('Elongation[{}]: {}' .format(j, e))
                        #imgcnt = np.zeros(imgcontours.shape[:2])
                        imag = img.copy()
                        imgcnt = cv2.drawContours(image=imgcnt, contours=[contour], contourIdx=-1, color=(255, 0, 0), thickness=cv2.FILLED)
                        imag = cv2.drawContours(image=imag, contours=[contour], contourIdx=-1, color=(0, 255, 0))
                        imgsave = cv2.drawContours(image=imgsave, contours=[contour], contourIdx=-1, color=(0, 255, 0))
                        std = standard_deviation(imgcnt)
                        std_f = 0.0
                        '''for k in std:
                                for m in k:
                                       std = m'''
                        print ('Standard Deviation[{}]: {}\n' .format(j, std))
                        name_image = 'Image {} / Contour {}' .format(i+1, j+1)
                        #cv2.imshow(name_image, imgcnt)
                        #cv2.imshow(name_image, imag)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        lista.append(i)
                        lista.append(j)
                        lista.append(a)
                        lista.append(round(l, 2))
                        lista.append(w)
                        lista.append(h)
                        lista.append(c)
                        lista.append(round(e, 2))
                        lista.append(round(std, 2))
                        list_features.append(a)
                        list_features.append(l)
                        list_features.append(w)
                        list_features.append(h)
                        list_features.append(c)
                        list_features.append(e)
                        list_features.append(std)
                        dictionary['data'].append(list_features)
                        list_features = []
                        j+=1
                save_image(imgsave, path_d[i])
                print ('Total of Contours: {}\n'.format(totalNumContours))
                #cv2.imshow(name_image, imag)
                #cv2.imshow(name_image, imgcnt)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        dictionary['image_name'] = np.array(list_name, dtype='|S23')

        dictionary['data'] = np.array(dictionary['data'])
        
        #Labels para 4 classes de falhas
        #Caso a quantidade de imagens seja mudada alterar as labels abaixo tambem
        dictionary['target'] = np.array([3, 0, 3, 0, 3, 3, 3, 3, 3, 0, 3, 2, 3, 3, 0, 3, 1, 3, 3, 3, 0, 0, 2, 0, 0, 2, 2, 3, 0, 2, 3, 0, 3, 3, 3, 0, 3, 3, 0, 2, 3,
                                         0, 3, 2, 3, 3, 0, 0, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3, 0, 2, 3, 3, 3, 0, 2, 3, 3, 3, 0, 3, 3, 3, 3, 3, 3, 1, 3, 0, 3,
                                         3, 3, 0, 0, 0, 0, 2, 1, 0, 1, 0, 0, 3, 0, 1, 3, 3, 3, 3, 0, 0, 1, 2, 3, 3, 3, 3, 3, 3, 0, 2, 3, 3, 3, 3, 3, 0, 3, 3, 3, 3, 3])

        #Labels para 2 classes de falhas
        '''dictionary['target'] = np.array([1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1,
                                         0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
                                         1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])'''

        label_names = dictionary['target_names']
        labels = dictionary['target']
        feature_names = dictionary['feature_names']
        features = dictionary['data']
        global global_features
        global_features = features
        global global_labels
        global_labels = labels
        global global_label_names
        global_label_names = label_names
        
        warnings.filterwarnings("ignore")
        
def stats():
        # Binariza saidas. Exemplo: para a label = 4 o resultado sera 001.
        lb = preprocessing.LabelBinarizer()
        labels_bin = lb.fit_transform(global_labels)
        n_classes = labels_bin.shape[1]
        n_test_contours = int(0.31*global_labels.size)
        
        train, test, train_labels, test_labels = train_test_split(global_features, labels_bin, test_size=n_test_contours, shuffle=False)

        for index, k, clf in zip(range(1,len(global_classifiers)+1), global_classifiers_names, global_classifiers):
                #Aprende a predizer cada classe contra as outras
                classifier = OneVsRestClassifier(clf)

                #Para os classificadores "Nearest Neighbors", "Gaussian Process", "Decision Tree", "Random Forest", "Neural Net", "Naive Bayes"
                '''label_score = classifier.fit(train, train_labels).predict_proba(test)'''
                #Para os classificadores "Linear SVM", "RBF SVM", "AdaBoost", "QDA"
                label_score = classifier.fit(train, train_labels).decision_function(test)

                preds = lb.inverse_transform(label_score)
                test_l = lb.inverse_transform(test_labels)

                print('  Real  ({}): {}' .format(k, test_l))

                print('Predicted({}): {}' .format(k, preds))

                #Calcula precisao, sensibilidade e medida-F
                print('Report({}):' .format(k))
                print(classification_report(test_l, preds, target_names = global_label_names))
                print('---------')

        print("Runtime: %s seconds" % (time.time() - start_time))
        print('---------')
        warnings.filterwarnings("ignore")
                
        
                
        
if __name__ == "__main__":
        start_time = time.time()
        main()
        stats()
