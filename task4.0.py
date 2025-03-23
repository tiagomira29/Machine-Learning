import numpy as np
import time

# tensorflow related
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# sklearn related
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# imblearn

# plot
import matplotlib.pyplot as plt

# files
x_train = np.load("Xtrain_Classification2.npy")/255
y_train = np.load("ytrain_Classification2.npy")
x_test = np.load("Xtest_Classification2.npy")/255

y_test = np.empty_like(y_train)

       
        
## Functions ##

def balanced_accuracy_metric(y_true,y_pred):
    # Initialize variables to store TP, FP, TN, and FN
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    # Calculate TP, FP, TN, FN
    for true, pred in zip(y_true, y_pred):
        if true == 1:
            if pred == 1:
                TP += 1
            else:
                FN += 1
        else:
            if pred == 1:
                FP += 1
            else:
                TN += 1
                
    sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    
    balanced_accuracy = (sensitivity + specificity) / 2
    
    return balanced_accuracy


# splits a dataset into two diferents groups
def split_function(x, y, val_percent, classes):
    
    class_x = []
    class_y = []
    cut = []
    #Spliting the classes
    for i,clas in enumerate(classes):
        class_x.append( x[y==clas] )
        class_y.append( y[y==clas] )
        cut.append( val_percent * len(class_x[i]) )
        #print(x[y==i])

    x_v = []
    y_v = []
    x_t = []
    y_t = []

    for i in range(len(classes)):
        #print(i, class_y[i], cut[i])
        x_v.append(class_x[i][:round(cut[i])])
        y_v.append(class_y[i][:round(cut[i])])
        x_t.append(class_x[i][round(cut[i]):])
        y_t.append(class_y[i][round(cut[i]):])

    #print(f"x_v: {x_v}\n y_v: {y_v}\n x_t: {x_t}\n y_t: {y_t}")
    x_v = [item for sublist in x_v for item in sublist]
    y_v = [item for sublist in y_v for item in sublist]
    x_t = [item for sublist in x_t for item in sublist]
    y_t = [item for sublist in y_t for item in sublist]

    return np.array(x_v), np.array(y_v), np.array(x_t), np.array(y_t)


# Quantidade de objetos de cada classe no training data
def classes_quantity(y_classes,n_labels):
    # count ocorrences of each class #
    y_count = np.zeros(n_labels)
    for y in y_classes:
        y_count[round(y)] += 1
            
    return y_count


# Cria imagens atraves de metodos
def image_creator(x_orig, y_orig, max_img, classes):
    
    x_orig = np.array(x_orig).reshape(-1,28, 28, 3)
    
    graus90 = np.empty_like(x_orig)
    graus180 = np.empty_like(x_orig)
    graus270 = np.empty_like(x_orig)
    flipv = np.empty_like(x_orig)
    fliph = np.empty_like(x_orig)
    
    for i,img in enumerate(x_orig):
        graus90[i] = np.rot90(img)
    for i,img in enumerate(graus90):
        graus180[i] = np.rot90(img)
    for i,img in enumerate(graus180):
        graus270[i] = np.rot90(img)
    for i,img in enumerate(x_orig):
        fliph[i] = np.fliplr(img)
        flipv[i] = np.flipud(img)
    
    new_x = np.vstack((graus90, graus180, graus270, fliph, flipv))
    new_y = np.concatenate((y_orig,y_orig,y_orig,y_orig,y_orig))
    print(new_x.shape, new_y.shape)
    total_images = len(new_x) + len(x_orig)
    if total_images > max_img:
        np.random.shuffle(new_x)
        new_x,new_y,trash_x,trash_y = split_function(new_x, new_y, (total_images - max_img) / len(new_x), classes)
    
    
    return np.concatenate((x_orig,new_x)), np.concatenate((y_orig,new_y))


# Fornece pesos a cada classe para compensar imbalanceamentos
def weight_implementation(y_array):
    # computing class weights #      image_creator + weight_implementation: balanced_accuracy = 77.8% + 94.2% -- total 85.7%
    labels_array = [0, 1, 2]
    weights_array = compute_class_weight(class_weight = 'balanced', classes = labels_array, y = y_array)
    print(weights_array)
    dic = {0: weights_array[0], 1: weights_array[1], 2: weights_array[2]}

    return dic


# Rede Neuronal que otimiza hiper parametros
def auto_opt_keras_model(kx_val, ky_val, kx_train, ky_train, dense_layers = [0, 1, 2], layer_sizes = [32, 64, 128], conv_layers = [1, 2, 3], learning_rates = [1e-2,1e-3,1e-4], dropouts = [0,0.05,0.1,0.3]):

    x_train_reshaped = np.array(kx_train).reshape(-1,28, 28, 3)
    kx_val = np.array(kx_val).reshape(-1,28, 28, 3)
    
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                for lr in learning_rates:
                    for dropout in dropouts:
                        NAME = f"{conv_layer}-conv-{layer_size}-nodes-{dense_layer}-dense-{lr}-lr-{dropout}-dropout-{int(time.time())}"
                        tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
                        print(NAME)
                        
                        model = Sequential()
                        
                        model.add(Conv2D(32, (3,3), activation='relu', input_shape=x_train_reshaped.shape[1:]))
                        model.add(MaxPooling2D(pool_size=(2,2)))
                        model.add(Dropout(dropout))
                        
                        for l in range(0,conv_layer-1,1):
                            model.add(Conv2D(64*(2**l), (3,3), activation='relu'))
                            model.add(MaxPooling2D(pool_size=(2,2)))
                            model.add(Dropout(dropout))
                            
                        model.add(Flatten())
                        for l in range(0,dense_layer,1):
                            model.add(Dense(layer_size, activation='relu'))
                            
                        model.add(Dense(3, activation='softmax'))
                        
                        model.compile(optimizer= Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
                        
                        model.fit(x_train_reshaped, to_categorical(ky_train), batch_size=32, epochs=20, validation_data=(kx_val,to_categorical(ky_val)), callbacks=[tensorboard])
                        y_pred = model.predict(np.array(x_test).reshape(-1,28, 28, 3))
                        np.save(f'logs\{NAME}\y_predict.npy', y_pred)


# Rede Neuronal binária para dividir data em dois (assume-se grande diferenca entre os datasets, pois nao se usa validation data)
def valless_binary_keras_model(vlx_train, vly_train):
    
    x_train_reshaped = np.array(vlx_train).reshape(-1,28, 28, 3)

    # NN model #
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=x_train_reshaped.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.summary()

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.summary()

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        optimizer= Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    early_stp = EarlyStopping(monitor = 'val_loss', patience = 50, mode= 'min', restore_best_weights = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=1e-6)
    # Fit
    model.fit(x_train_reshaped, vly_train, batch_size=32, epochs=10)
    # Fit with class weighs
    # model.fit(x_train_reshaped, to_categorical(ky_train), batch_size=32, epochs=50, validation_data=(kx_val,to_categorical(ky_val)), class_weight=dict, callbacks=[early_stp])

    predictions = model.predict(x_train_reshaped)
    pred_test = model.predict(np.array(x_test).reshape(-1,28, 28, 3))
    
    return predictions, pred_test


# main #

y_bin = np.load("ytrain_Classification2.npy")
y_bin = [0 if value in (0, 1, 2) else 1 for value in y_bin]
y_bin = np.array(y_bin)
print(y_bin.shape)

# Separate data into 2 diferent data sets
y_bin_predict, y_test_bin = valless_binary_keras_model(x_train, y_bin)
y_bin_predict = np.array([round(value[0]) for value in y_bin_predict])
x_train1 = x_train[y_bin_predict==0]
x_train2 = x_train[y_bin_predict==1]
y_train1 = y_train[y_bin_predict==0]
y_train2 = y_train[y_bin_predict==1]
y_train2 = np.array([round(label - 3) for label in y_train2])

y_test_bin = np.array([round(value[0]) for value in y_test_bin])
x_test1 = x_test[y_test_bin==0]
x_test2 = x_test[y_test_bin==1]

y_test1 = np.empty_like(y_train1)
y_test2 = np.empty_like(y_train2)

## Classes ##
class CustomModelCheckpoint1(tf.keras.callbacks.Callback):
    def __init__(self, monitor, verbose=0, save_best_only=False):
        super(CustomModelCheckpoint1, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.y_test = None
        self.best_score = None
        self.epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        global y_test1
        epoch += 1
        y_true = self.validation_data[1]  # Replace with your actual validation labels
        y_pred = self.model.predict(self.validation_data[0])
        
        score = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        print(score)
        if best_score == None:
            best_score = score
        elif best_score < score:
            best_score = score
            self.y_test = self.model.predict(np.array(x_test).reshape(-1,28, 28, 3))

        if epoch==50:
            y_test1 =  self.y_test
            for pred in y_test1:
                y_test1.append(np.argmax(pred))
            y_test1 = np.array(y_test1)

class CustomModelCheckpoint2(Callback):
    def __init__(self, monitor, verbose=0, save_best_only=False):
        super(CustomModelCheckpoint2, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.y_test = None
        self.best_score = None
        self.epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        global y_test2
        epoch += 1
        y_true = self.validation_data[1]  # Replace with your actual validation labels
        y_pred = self.model.predict(self.validation_data[0])
        
        score = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        print(score)
        if best_score == None:
            best_score = score
        elif best_score < score:
            best_score = score
            self.y_test = self.model.predict(np.array(x_test).reshape(-1,28, 28, 3))

        if epoch==50:
            y_test2 =  self.y_test
            for pred in y_test2:
                y_test2.append(np.argmax(pred))
            y_test2 = np.array(y_test2)


# Rede Neuronal
def keras_model(kx_val, ky_val, kx_train, ky_train, output_size, kx_test1):
    
    x_train_reshaped = np.array(kx_train).reshape(-1,28, 28, 3)
    kx_val = np.array(kx_val).reshape(-1,28, 28, 3)

    # NN model #
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=x_train_reshaped.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.summary()

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.summary()

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    if output_size > 2:
        model.add(Dense(output_size))
        model.add(Activation('softmax'))
    else:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

    model.compile(
        optimizer= Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(filepath='best_model_weights.h5', monitor='val_balanced_accuracy_metric',save_best_only=True,save_weights_only=True,mode='auto',verbose=0)
    # Callbacks
    early_stp = EarlyStopping(monitor = 'val_loss', patience = 5, mode= 'min', restore_best_weights = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=1e-6)
    # Fit
    model.fit(x_train_reshaped, to_categorical(ky_train), batch_size=32, epochs=50, validation_data=(kx_val,to_categorical(ky_val)), callbacks=[early_stp,CustomModelCheckpoint2])
    # Fit with class weighs
    # model.fit(x_train_reshaped, to_categorical(ky_train), batch_size=32, epochs=50, validation_data=(kx_val,to_categorical(ky_val)), class_weight=dict, callbacks=[early_stp])

    predictions = model.predict(kx_val)
    final_y_test1 = model.predict(np.array(kx_test1).reshape(-1,28, 28, 3))
    
    return predictions, final_y_test1


# Rede Neuronal
def keras_model_weighted(kx_val, ky_val, kx_train, ky_train, output_size, dic, kx_test2):
    
    x_train_reshaped = np.array(kx_train).reshape(-1,28, 28, 3)
    kx_val = np.array(kx_val).reshape(-1,28, 28, 3)

    # NN model #
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=x_train_reshaped.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.summary()

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.summary()

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    if output_size > 2:
        model.add(Dense(output_size))
        model.add(Activation('softmax'))
    else:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

    model.compile(
        optimizer= Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(filepath='best_model_weights.h5', monitor='val_balanced_accuracy_metric',save_best_only=True,save_weights_only=True,mode='auto',verbose=0)

    # Callbacks
    early_stp = EarlyStopping(monitor = 'val_loss', patience = 10, mode= 'min', restore_best_weights = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, min_lr=1e-6)
    # Fit with class weighs
    model.fit(x_train_reshaped, to_categorical(ky_train), batch_size=32, epochs=50, validation_data=(kx_val,to_categorical(ky_val)), class_weight=dic, callbacks=[early_stp,CustomModelCheckpoint1])

    predictions = model.predict(kx_val)
    final_y_test2 = model.predict(np.array(kx_test2).reshape(-1,28, 28, 3))
    
    return predictions, final_y_test2

# splitting data into training data and validation data #
x_val1, y_val1, x_train1, y_train1 = split_function(x_train1, y_train1, 0.2, [0,1,2])
x_val2, y_val2, x_train2, y_train2 = split_function(x_train2, y_train2, 0.2, [0,1,2])

class_quantities1 = classes_quantity(y_train1,3)
class_quantities2 = classes_quantity(y_train2,3)
print(class_quantities1, class_quantities2)
new_x1 = []
new_y1 = []
new_x2 = []
new_y2 = []
# Data augmentation
for i in range(0,len(class_quantities1)):
    temp_x, temp_y = image_creator(x_train1[y_train1==i],y_train1[y_train1==i],np.max(class_quantities1), [0,1,2])
    new_x1.append(temp_x)
    new_y1.append(temp_y)
x_train1_resampled = np.concatenate(new_x1,axis=0)
y_train1_resampled = np.concatenate(new_y1,axis=0)
for i in range(0,len(class_quantities2)):
    temp_x, temp_y = image_creator(x_train2[y_train2==i],y_train2[y_train2==i],np.max(class_quantities2), [0,1,2])
    new_x2.append(temp_x)
    new_y2.append(temp_y)
x_train2_resampled = np.concatenate(new_x2,axis=0)
y_train2_resampled = np.concatenate(new_y2,axis=0)

y_train1_resampled = np.array([np.round(value) for value in y_train1_resampled])
y_train2_resampled = np.array([np.round(value) for value in y_train2_resampled])

class_quantities1 = classes_quantity(y_train1,3)
print(class_quantities1)

dic = weight_implementation(y_train1_resampled)
# Prediction com NN
#auto_opt_keras_model(x_val1,y_val1,x_train1_resampled,y_train1_resampled,dense_layers=[1],learning_rates=[1e-4])
y_pred1, final_y_test1 = keras_model_weighted(x_val1,y_val1,x_train1_resampled,y_train1_resampled, 3, dic, x_test1)
y_pred2, final_y_test2 = keras_model(x_val2,y_val2,x_train2_resampled,y_train2_resampled, 3, x_test2)

y_predict1 = []
y_predict2 = []

for pred in y_pred1:
    y_predict1.append(np.argmax(pred))
y_predict1 = np.array(y_predict1)
for pred in y_pred2:
    y_predict2.append(np.argmax(pred) + 3)
y_predict2 = np.array(y_predict2)
y_val1 = [np.round(value) for value in y_val1]
y_val2 = [np.round(value) + 3 for value in y_val2]
y_val = np.concatenate((y_val1,y_val2))
y_predict = np.concatenate((y_predict1,y_predict2))
# np.savetxt("y_predict.txt", y_predict, delimiter=",", fmt="%d")
# np.savetxt("y_val.txt", y_val, delimiter=",", fmt="%d")
ba = balanced_accuracy_score(y_true=y_val, y_pred=y_predict)
ba1 = balanced_accuracy_score(y_true=y_val1, y_pred=y_predict1)
ba2 = balanced_accuracy_score(y_true=y_val2, y_pred=y_predict2)

fy_test1 = []
fy_test2 = []
for pred in final_y_test1:
    fy_test1.append(np.argmax(pred))
fy_test1 = np.array(fy_test1)
for pred in final_y_test2:
    fy_test2.append(np.argmax(pred) + 3)
fy_test2 = np.array(fy_test2)

y_test_predict = []
i = 0
j = 0
for bin in y_test_bin:
    if np.round(bin) == 0:
        y_test_predict.append(fy_test1[i])
        i += 1
    else:
        y_test_predict.append(fy_test2[j])
        j += 1
y_test_predict = np.array(y_test_predict)
y_test_predict = np.array(y_test_predict).reshape(-1,)
np.save("y_test_predict", y_test_predict)


print(f"total balanced accuracy: {ba}")
print(f"balanced accuracy data set 1: {ba1}")
print(f"balanced accuracy data set 2: {ba2}")
print(f"classification report both data sets:\n{classification_report(y_val, y_predict)}")
print(f"classification report data set 1:\n{classification_report(y_val1, y_predict1)}")
print(f"classification report data set 2:\n{classification_report(y_val2, y_predict2)}")

