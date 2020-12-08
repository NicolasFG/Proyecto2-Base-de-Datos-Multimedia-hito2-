from rtree import index
from os import listdir
from os.path import isfile, isdir
import cv2
from rtree import index
import face_recognition
import heapq
import json
from os import remove
import os.path as path
from time import time

#Funciones de distancia
def Manhattan(ValorA,ValorB):
    return sum([abs(x1 - x2) for (x1, x2) in zip(ValorA, ValorB)])

def Euclidiana(ValorA,ValorB):
    return (sum([(x1 - x2)**2 for (x1, x2) in zip(ValorA, ValorB)]))**(0.5)

#Transformar un array en una tupla en la forma correcta para el Btree
def TransformarArrayATupla(A):
    return tuple(A+A)


#Convierte el valor retornado de face_recognition a una lista
def ConvertirLista(Lista):
    ListaResultado=[]
    for i in Lista:
        ListaResultado.append(i)
    return ListaResultado


##Funcion usada antes de todo para crear los vectores caracteristicos de las imagenes y guardarlos
def CreacionVectorCaracteristico():
    i=0
    rutaDeLasImagenes="./dataset/"
    vectores={}
    for carpeta in listdir(rutaDeLasImagenes):
        for objeto in listdir(rutaDeLasImagenes+carpeta+'/'):
            print(objeto)
            print(i)
            i+=1

            ObjetoImagen = face_recognition.load_image_file(rutaDeLasImagenes+carpeta+'/'+objeto)
            ObjetoImagen_encoding = face_recognition.face_encodings(ObjetoImagen)

            if len(ObjetoImagen_encoding)>0:
                vectores[objeto]=list(ObjetoImagen_encoding[0])

    with open('vectors.json', 'w') as file:
        json.dump(vectores, file)


#Funcion KnnRtree, usando la libreria implementada en python
def KnnRtree(Q,k):

    #Eliminacion del indice previo para crear uno nuevo
    if path.exists('rtree_index.data'):
        remove('rtree_index.data')
    if path.exists('rtree_index.index'):
        remove('rtree_index.index')
    

    #Construyo el indice el rtree por default
    rtree = index.Property()
    

    #Dimension del Rtree
    rtree.dimension = 256

    #Capacidad del buffering
    rtree.buffering_capacity = 23


    #Para guardarlo en disco y que sea mas optimo el espacio reservado, ya que no se guarda en la RAM.
    #La libreria permite crear dos extenciones, Una es data y la otra es index.
    
    rtree.dat_extension = 'data'
    rtree.idx_extension = 'index'

    #Inicialiso el rtree de acurdoo a las dos extenciones creadas
    NuevoIndex = index.Index('rtree_index',properties=rtree)


    #Se carga el vector caracteristico
    with open('vectors.json',errors='ignore',encoding='utf8') as contenido:
        datos=json.load(contenido)



    #Se crea una lista en donde se guardara los objetos cargados al recorer datos
    keys=[]
    id=0
    for i in datos:
        keys.append(i)

        #Se inserta dentro Rtree un id conjunto a su repesctiva tupla
        NuevoIndex.insert(id,TransformarArrayATupla(datos[i]))
        id+=1

    tiempo_inicial = time()

    #Se realiza la busqueda en el indice del Rtree
    Tupla = TransformarArrayATupla(Q)

    ListaResultado = list(NuevoIndex.nearest(coordinates=Tupla, num_results=k))
    tiempo_final = time()

    tiempo= tiempo_final - tiempo_inicial

    print("Tiempo de rtree ",tiempo)

    print("Imagenes próximas: ", ListaResultado)
    for i in ListaResultado:
        print(keys[i])




def KnnSecuencial(Q,k):
    ##Busqueda SECUENCIAL tomando las 2 distancias pedidas y al final mostrar ambos resultados
    tiempo_inicial = time()
   
    with open('vectors.json',errors='ignore',encoding='utf8') as contenido:
        datos=json.load(contenido)

    #Convierto en una lista el valor obtenido de face_recognition
    QLista=ConvertirLista(Q)
    
    #Listas de distancias
    DistanciaMan=[]
    DistanciaEuc=[]
    
    #Recorro el vector caracteristico de toda mi dataset
    for i in datos:

        #Obtengo el valor de la distancia Euclidiana y Manhattan
        eu = Euclidiana(QLista,datos[i])
        man = Manhattan(QLista,datos[i])

        
        #Guardo la tupla de distancia obtenida y nombre del objeto en un priority queue
        

        heapq.heappush(DistanciaMan,(-man,i))
        heapq.heappush(DistanciaEuc,(-eu,i))

        #Condicional para trabajar con los k elementos que pasaremos como parametro
        if len(DistanciaMan)>k:
                heapq.heappop(DistanciaMan)
        if len(DistanciaEuc)>k:
                heapq.heappop(DistanciaEuc)

    print(DistanciaEuc)
    print("\n")
    print(DistanciaMan)


    tiempo_final = time()
    tiempo= tiempo_final - tiempo_inicial

    print("DIstancia Manhattan")
    for i in DistanciaMan:
        print(i[0],end=' ')
        print(i[1])
    print("Distancia Euclidiana")
    for i in DistanciaEuc:
        print(i[0],end=' ')
        print(i[1])
    print("Tiempo Knn secuencial ",tiempo)



def AccederAImagenEnCadaCarpeta(cadena):
    separador = "0"
    separado_por_espacios = cadena.split(separador)
    string=separado_por_espacios[0]
    result = string.rstrip('_')
    return result+"/"





##Imagen para la prueba
aux = face_recognition.load_image_file("foto3.jpg")
aux_1=face_recognition.face_encodings(aux)[0]

KnnSecuencial(ConvertirLista(aux_1),10)
KnnRtree(ConvertirLista(aux_1),2)

#CreacionVectorCaracteristico()