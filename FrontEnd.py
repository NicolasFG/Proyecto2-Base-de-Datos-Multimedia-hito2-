import face_recognition
from flask import Flask, jsonify, request, redirect, render_template
from rtree import index
from os import listdir
from os.path import isfile, isdir
import cv2
from rtree import index
import heapq
import json
from os import remove
import os.path as path

app = Flask(__name__)

ExtensionesPermitidas = {'png', 'jpg', 'jpeg', 'gif'}



#Adecuar cada imagen a los archivos permitidos
def PermitirArchivos(Archivo):
    return '.' in Archivo and \
           Archivo.rsplit('.', 1)[1].lower() in ExtensionesPermitidas


#Para poder acceder correctamente a la imagen dentro de cada carpeta
def AccederAImagenEnCadaCarpeta(Cadena):
    Limitador = "0"
    SeparadoEspacios = Cadena.split(Limitador)
    CadenaObtendia = SeparadoEspacios[0]
    resultado = CadenaObtendia.rstrip('_')
    return resultado+"/"



@app.route('/', methods=['GET', 'POST'])
#Carga de la imagen
def CargarImagen():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        number = request.form['k']

        if file.filename == '':
            return redirect(request.url)

        if file and PermitirArchivos(file.filename) and number:
            return DeteccionDeCaras(file,number)
    return render_template("Inicio.html")



def TransformarALista(Lista):
    ListaResultante=[]
    for i in Lista:
        ListaResultante.append(i)
    return ListaResultante



def TransformarArrayATupla(Lista):
    return tuple(Lista+Lista)



def KnnRtree(Q,k):

    if path.exists('rtree_index.data'):
        remove('rtree_index.data')
    if path.exists('rtree_index.index'):
        remove('rtree_index.index')


    rtree = index.Property()
    rtree.dimension = 256
    rtree.buffering_capacity = 23
    rtree.dat_extension = 'data'
    rtree.idx_extension = 'index'


    NuevoIndex = index.Index('rtree_index',properties=rtree)

    with open('vectors.json',errors='ignore',encoding='utf8') as contenido:
        datos=json.load(contenido)
    keys=[]
    id=0
    for i in datos:
        keys.append(i)
        NuevoIndex.insert(id,TransformarArrayATupla(datos[i]))
        id+=1

    Tupla = TransformarArrayATupla(Q)

    ListaResultado = list(NuevoIndex.nearest(coordinates=Tupla, num_results=k))
    print("Imagenes próximas: ", ListaResultado)
    Resultado=[]
    for i in ListaResultado:
        Resultado.append(keys[i])
        print(keys[i])
    return Resultado




def DeteccionDeCaras(Archivo, Numero):

    imagen = face_recognition.load_image_file(Archivo)

    FaceEncodings = face_recognition.face_encodings(imagen)

    k=int(Numero)

    imagenes = KnnRtree(TransformarALista(FaceEncodings[0]),k)

    for i in range(0,len(imagenes)):
        imagenes[i]=AccederAImagenEnCadaCarpeta(imagenes[i])+imagenes[i]
        print(imagenes[i])
    return render_template("Resultados.html",r=imagenes)



if __name__ == "__main__":
    print("Iniciando el programa")
    app.run(host='0.0.0.0', port=5001, debug=True)
