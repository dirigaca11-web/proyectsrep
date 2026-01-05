

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam





def load_data(path, width, height, batch, directory):
    
    labels = pd.read_csv( path)
    train_df, val_df = train_test_split( labels,test_size=0.25,random_state=12345)
    
    #agregamos aumento de datos en nuestro generador 
    train_datagen = ImageDataGenerator(
        rescale=1./255.,
        horizontal_flip=True,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
        )

    #crearemos el generador del entrenamiento desde el dataframe de etiquetas
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=directory,  
        target_size=(width,height),   
        batch_size=batch,
        class_mode='raw',     
        seed=12345,
        x_col='file_name',
        y_col='real_age',
        subset='training'
    )

    
    return train_generator

def load_test(path, width, height, batch,directory):
    
    labels = pd.read_csv( path)
    train_df, val_df = train_test_split( labels,test_size=0.25,random_state=12345)
    val_datagen = ImageDataGenerator(rescale=1./255.)
    
    
    #crearemos el generador de la validación desde el dataframe de etiquetas
    val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=directory,
    target_size=(width,height),   
    batch_size= batch,
    class_mode='raw',     
    seed=12345,
    x_col='file_name',
    y_col='real_age',
    )

    return val_generator
    

def create_model(input_shape):
    backbone = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    
            
    # entrenemos la parte de clasificación
    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='relu'))
    
    optimizer = Adam(learning_rate=0.00005 )
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
 
    
    return model
    

def train_model(model, train_data, test_data, batch_size=None, epochs=3,
                steps_per_epoch=None, validation_steps=None):

    
    model.fit(train_data, 
              validation_data=test_data,
              batch_size=batch_size, epochs=epochs,
              verbose=2)


    return model
    
width=224
height=224
input_shape = (width, height, 3)
batch=16

path='C:/Users/drgarciacabo/Music/tripleten/DS/sprint 17 visión artificial/proyecto sprint 17/labels.csv'
direct='C:/Users/drgarciacabo/Music/tripleten/DS/sprint 17 visión artificial/proyecto sprint 17/final_files/'


train = load_data(path=path, width=width,height=height,batch=batch, directory=direct ) 
test = load_test(path=path, width=width,height=height,batch=batch , directory=direct )
model = create_model(input_shape)
model = train_model(model, train, test)
