#import Liberaries 
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras.utils
from keras import utils as np_utils
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from timeit import default_timer as timer
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#Add the path of training and validation dataset
train_data_dir = '/content/drive/My Drive/Major_Project/Dataset/train'
validation_data_dir = '/content/drive/My Drive/Major_Project/Dataset/val'

# Change to fit hardware
batch_size = 128
#Total  7 categories of emotion ('Angry','Disgust','Fear','Happy','Sad','Surprize') 
num_classes = 7
img_rows, img_cols = 48, 48
#use Datagenerators to load the data
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      shear_range=0.3,
      zoom_range=0.3,
      width_shift_range=0.4,
      height_shift_range=0.4,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        color_mode = 'grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode = 'grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)

#use the mini xception model
def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
    regularization = l2(l2_regularization)

    # base
    img_input = Input(input_shape)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization,
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # module 1
    residual = Conv2D(16, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(16, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 2
    residual = Conv2D(32, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(32, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 3
    residual = Conv2D(64, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    # module 4
    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same',
                        kernel_regularizer=regularization,
                        use_bias=False)(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    x = Conv2D(num_classes, (3, 3),
               # kernel_regularizer=regularization,
               padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    output = Activation('softmax', name='predictions')(x)

    model = Model(img_input, output)
    return model
    
    # model parameters/compilation
model = mini_XCEPTION((48,48,1), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.compile(loss = 'categorical_crossentropy',optimizer = Adam(lr=0.001),metrics = ['accuracy'])
#model.summary()

def train(model,
         train_generator,
         validation_generator,
         max_epochs_stop=3,
         n_epochs=20,
         print_every=1):
  
  earlystop = EarlyStopping(monitor = 'val_loss',min_delta = 0, patience = 3,verbose = 1,restore_best_weights = True)
    
  reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, min_delta = 0.0001)

  trained_models_path = '/content/drive/My Drive/Major_Project/Dataset/trainmodels/'
  model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
  model_checkpoint = ModelCheckpoint(model_names, 'val_loss', verbose=1,save_best_only=True)
  
  callbacks = [model_checkpoint,earlystop, reduce_lr]

  
  # Early stopping intialization
  epochs_no_improve = 0
  valid_loss_min = np.Inf

  valid_max_acc = 0
  history = []

  nb_train_samples = 29497
  nb_validation_samples = 7352

  # Number of epochs already trained (if using loaded in model weights)
  #try:
  #    print(f'Model has been trained for: {model.epochs} epochs.\n')
  #except:
  #    model.epochs = 0
  #    print(f'Starting Training from Scratch.\n')

  overall_start = timer()


  #model_checkpoint = ModelCheckpoint(save_file_name, 'val_loss', verbose=1,save_best_only=True)

  history = model.fit_generator(train_generator,
                                steps_per_epoch = nb_train_samples // batch_size,
                                epochs = n_epochs,
                                callbacks = callbacks,
                                validation_data = validation_generator,
                                validation_steps = nb_validation_samples // batch_size)

    
  return model,history  
  
  m,h = train(model,train_generator,validation_generator,n_epochs=20)
