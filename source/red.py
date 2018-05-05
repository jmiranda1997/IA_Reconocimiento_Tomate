#!/usr/bin/python
from fann2 import libfann
from glob import glob
from PIL import Image

def evaluar(imagen):
	#Cargamos la red neuronal ya entrenada
	ann = libfann.neural_net()
	ann.create_from_file("rna2_3.net")
	#Abrimos la imágen
	nueva  = Image.open(imagen)
	#Redimensionamos la imágen a 50x50px y lo pasamos a una lista, donde cada elemto es un píxel [r,g,b]
	nueva = nueva.resize((50,50))
	arreglo = list(nueva.getdata())
	tmep = []
	#Vectorizamos completamente el arreglo de la imágen
	for x in arreglo:
		tmep.append(x[0])#combinamos las posisiones
		tmep.append(x[1])
		tmep.append(x[2])
	tmep.append(50)#le decimos el tamaño de la imagen
	return ann.run(tmep)#analizamos el resultado
