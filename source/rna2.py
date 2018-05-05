#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image #biblioteca para abrir y escalar imagenes
from glob import glob #bibilioteca utilizada para iterar entre los archivos de una biblioteca
from numpy import array 
from fann2 import libfann # biblioteca para entrenar la RNA  

imagenes = [] #almacena las iamgenes de entrada
esperados = [] #almacena los resultados esperados


def getdata():
	"""Funcion encargada de obtenes los datos para el entrenamientos ubicados en ./Resources/recortadas"""
	#importamos las variables globales
	global imagenes 
	global esperados
	#Primero obtenemos los errores y esperamos que nos regrese todas o la mayoria de banderas levantadas
	for imagen in glob("./Resources/recortadas/error/*.jpg"):
		nueva  = Image.open(imagen) #abrimos la imagen 
		nueva = nueva.resize((50,50)) #la escalamos
		arreglo = list(nueva.getdata()) #creamos una lista por ser RGB la lista contine arreglos de pixeles con valores RGB por lo caual los separamos
		esperados.append([1,1,1]) #agregamos los esperados
		tmep = []
		for x in arreglo:
			tmep.append(x[0])
			tmep.append(x[1])
			tmep.append(x[2])
			pass
		tmep.append(50)
		imagenes.append(tmep) # lo agregamos a la lista de imagenes 
		pass
	#luego el mismo procedimiento para las etapas 1,2 y 3
	for imagen in glob("./Resources/recortadas/1/*.jpg"):
		nueva  = Image.open(imagen)
		nueva = nueva.resize((50,50))
		arreglo = list(nueva.getdata())
		esperados.append([0,0,1])
		tmep = []
		for x in arreglo:
			tmep.append(x[0])
			tmep.append(x[1])
			tmep.append(x[2])
			pass
		tmep.append(50)
		imagenes.append(tmep)
		pass
	for imagen in glob("./Resources/recortadas/2/*.jpg"):
		nueva  = Image.open(imagen)
		nueva = nueva.resize((50,50))
		arreglo = list(nueva.getdata())
		esperados.append([0,1,0])
		tmep = []
		for x in arreglo:
			tmep.append(x[0])
			tmep.append(x[1])
			tmep.append(x[2])
			pass
		tmep.append(50)
		imagenes.append(tmep)
		pass
	for imagen in glob("./Resources/recortadas/3/*.jpg"):
		nueva  = Image.open(imagen)
		nueva = nueva.resize((50,50))
		arreglo = list(nueva.getdata())
		esperados.append([1,0,0])
		tmep = []
		for x in arreglo:
			tmep.append(x[0])
			tmep.append(x[1])
			tmep.append(x[2])
			pass
		tmep.append(50)
		imagenes.append(tmep)
		pass


getdata() #ejecutamos lo funcion para obtener los datos




rango_de_conexion = 1 #nos dice qeu tipo de red se usa y esta es multi capa
variable_entrenamiento = .01 #constante de aprendizaje de la RNA

error_minimo = 0.0001 #definimos el error minimo al que se quiere llegar
iteraciones_maximas = 1000 #las iteraciones maximas al entrenar
iteraciones_reporte = 50 #numero de iteraciones por reporte

red = libfann.neural_net() # creamos la red FANN
red.create_sparse_array(rango_de_conexion, (7501, 2500,3))#le decimo que tipo de red y las neuronas por capa 
red.set_learning_rate(variable_entrenamiento) #le decimos que variable de entrenamiento se desea
red.set_activation_function_output(libfann.SIGMOID_SYMMETRIC) #la funcion a utilizar
datos = libfann.training_data() #creamos la variable para los datos

	
datos.set_train_data(imagenes,esperados) #cargamos los datos de entrenamiento y los esperados

red.train_on_data(datos, iteraciones_maximas, iteraciones_reporte, error_minimo) # entrenamos la red

red.save("rna2.net") # se guardan los resultados