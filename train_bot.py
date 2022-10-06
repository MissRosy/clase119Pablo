# Biblioteca de preprocesamiento de datos de texto
from operator import index
import nltk
nltk.download('punkt')

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

words=[]
classes = []
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)



def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words

for intent in intents['intents']:
    
        # Agregar todas las palabras de los patrones a una lista
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                      
            word_tags_list.append((pattern_word, intent['tag']))
        # Agregar todas las etiquetas a la lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)

print(stem_words)
print(word_tags_list[0]) 
print(classes)   

# Crear un corpus de palabras para el chatbot
def create_bot_corpus(stem_words, classes):

    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

print(stem_words)
print(classes)

# Crear una bolsa de palabras

training_data=[]
classNumbs=len(classes)
labels=[0]*classNumbs

for i in word_tags_list:
    bagW=[]
    Frases=i[0]
    for j in Frases:
        indexFra=Frases.index(j)
        Framin=stemmer.stem(j.lower())
        Frases[indexFra]=j
    for j in Frases:
            if j in Frases:
                bagW.append(1)
            else:
                bagW.append(0)
    print(bagW)
    labels_encoding = list(labels) 
    tag = i[1] 
    tag_index = classes.index(tag)  
    labels_encoding[tag_index] = 1  
    training_data.append([bagW, labels_encoding])

print(training_data[0])

# Crear datos de entrenamiento
def prePossed_train_data(training_data):
    conArr=np.array(training_data,dtype=object)
    train_x=list(conArr[:,0])
    train_y=list(conArr[:,1])
    print(train_x[0])
    print(train_y[0])
    return train_x,train_y 

train_x, train_y = prePossed_train_data(training_data)   