"""
**autor:** @SamyConejo

*   **Tema:** Prototipo de sistema móvil para reconocimiento facial aplicando Redes Neuronales Convolucionales y Transferencia de Aprendizaje.

***Proyecto Final***
1.  Configuración, entrenamiento y evaluación de la RNC MobileNetV2 para aplicar Transfer Learning y exportar modelos aplicando Stratified Cross Validation.
"""

import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, \
    classification_report, PrecisionRecallDisplay, roc_curve, roc_auc_score, precision_recall_curve, auc
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import optimizers
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
from statistics import mean, stdev
import matplotlib.pyplot as plt
import itertools


# our top custom layers.
def lw(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(256, activation='relu')(top_model)
    top_model = Dense(NUM_CLASSES, activation='softmax')(top_model)
    return top_model


# method to manage history between folds (aux method to use with no early stopping)
def manage_history(data):
    aux_loss = []
    aux_val_loss = []
    for h in data:
        temp_loss = h.history['loss']
        aux_loss.append(temp_loss)

        temp_loss = h.history['val_loss']
        aux_val_loss.append(temp_loss)

    aux_loss = np.array(aux_loss)
    loss_mean = np.mean(aux_loss, axis=0)

    aux_val_loss = np.array(aux_val_loss)
    val_loss_mean = np.mean(aux_val_loss, axis=0)

    loss_epoch_plot(loss_mean, val_loss_mean)


# method to build loss versus epoch plot
def loss_epoch_plot(loss, val_loss):
    plt.plot(loss, label='Training')
    plt.plot(val_loss, label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel('Mean of Loss')
    plt.legend()
    plt.savefig('loss_epoch' + '.png', dpi=300, bbox_inches='tight')
    plt.show()


# method to build precision vs recall plot
def prec_recall_plot(y_test, y_pred):
    PrecisionRecallDisplay.from_predictions(y_test, y_pred, name='Precision-Recall')
    plt.title('Precision/Recall ')
    plt.savefig('prec_recall' + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


# method to build confusion matrix
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', filename='n', cmap=None, normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('matrix_' + str(filename), dpi=300, bbox_inches='tight');
    plt.show()


# aux method to save trained model of each fold
def get_model_name(k):
    return 'model_' + str(k) + '.h5'


def metrics(y_true, y_pred, fold):
    print('Metrics Model ', fold)
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='macro')
    f1Score = f1_score(y_true, y_pred, average='macro')
    print("Accuracy  : {}".format(acc))
    print("Precision : {}".format(pre))
    print("F1 Score  : {}".format(f1Score))
    c_matrix = confusion_matrix(y_true, y_pred)

    print(c_matrix)
    report = classification_report(y_true, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   target_names=LABELS)
    print(report)
    return acc, pre, f1Score, c_matrix


if __name__ == '__main__':

    IMAGE_SIZE = 224  # image size
    BATCH_SIZE = 64  # number of images we are inputting into the neural network at once.
    EPOCHS = 100  # number of epochs to train
    NUM_CLASSES = 10  # we have 10 people to predict

    # MobileNetV2, not including top layers to enable Transfer Learning
    MobileNet = MobileNetV2(weights='imagenet',
                            include_top=False,
                            input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # We freeze layers to train just head layers
    for layer in MobileNet.layers:
        layer.trainable = False

    # print layers to check freeze
    for (i, layer) in enumerate(MobileNet.layers):
        print(str(i) + " " + layer.__class__.__name__, layer.trainable)

    # lists to save performance between folds
    accuracy = []
    precision_list = []
    recall_list = []
    y_true_matrix = []
    y_pred_matrix = []
    aux_history = []
    y_pred_proba = []
    auc_list = []

    # path to dataset (train and validation merged)
    image_dir = '/Users/samcn96/Desktop/DATA_MINING/data/'
    # path to saved models over ten folds
    save_dir = '/Users/samcn96/Desktop/DATA_MINING/saved_models/'
    # path to testing data folder
    test_path = '/Users/samcn96/Desktop/DATA_MINING/test/'

    LABELS = ["Elvis", "Fausto", "Ale", 'Jenifer', 'Malki', 'Maru', 'Nayta', 'Roberto', 'Samy', 'Sulay']

    # Number of images on dataset (train, validation)
    INSTANCES = 50522
    # count of folds
    fold_count = 1

    # config Stratified Cross Validation
    skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    train_data = pd.read_csv('images.csv')
    Y = train_data[['label']]

    # manage input data, here we rescale images.
    idg = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # structure to use Stratified Cross Validation
    for train_index, val_index in skf.split(np.zeros(INSTANCES), Y):

        training_data = train_data.iloc[train_index]
        validation_data = train_data.iloc[val_index]

        # reads image files given by CV from dataset to get training data
        train_generator = idg.flow_from_dataframe(training_data, directory=image_dir,
                                                  x_col="filename", y_col="label",
                                                  class_mode="categorical", shuffle=True,
                                                  target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                  batch_size=BATCH_SIZE, )

        # reads image files given by CV from dataset to get validation data
        val_generator = idg.flow_from_dataframe(validation_data, directory=image_dir,
                                                x_col="filename", y_col="label",
                                                class_mode="categorical", shuffle=True,
                                                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                batch_size=BATCH_SIZE, )

        # we use callbacks to control training process
        checkpoint = ModelCheckpoint(save_dir + get_model_name(fold_count),
                                     monitor="val_accuracy",
                                     mode="max",
                                     save_best_only=True,
                                     period=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',
                                      factor=0.5,
                                      patience=2,
                                      verbose=1,
                                      mode='max',
                                      min_lr=0.000001)

        earlystop = EarlyStopping(monitor='val_accuracy',
                                  patience=5,
                                  verbose=1,
                                  mode='max',
                                  restore_best_weights=True)

        callbacks = [earlystop, reduce_lr, checkpoint]

        # instace the model
        FC_Head = lw(MobileNet, NUM_CLASSES)
        model = Model(inputs=MobileNet.input, outputs=FC_Head)

        # compile and fit model with our configuration, we use SGD
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # starts training
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            callbacks=callbacks,
            validation_data=val_generator
        )

        # Mean Accuracy
        print('Mean Training Accuracy ', mean(history.history['accuracy']))
        print('Mean Validation Acurracy ', mean(history.history['val_accuracy']))

        # plot loss vs epoch per fold
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')

        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('loss_' + str(fold_count), dpi=300)
        plt.close()

        # plot accuracy vs epoch
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')

        plt.xlabel("Epoch")
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('accuracy_' + str(fold_count), dpi=300)
        plt.close()

        # test on testing folder
        test_generator = idg.flow_from_directory(
            test_path,
            target_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            class_mode=None,
            shuffle=False)

        predictions = model.predict(test_generator, verbose=1)

        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes
        y_proba = predictions

        # plot confusion matrix per fold
        testAcc, testPrec, testFScore, cm = metrics(y_true, y_pred, fold_count)
        plot_confusion_matrix(cm, LABELS, filename=str(fold_count))

        # save metrics in global lists
        accuracy.append(accuracy_score(y_true, y_pred))
        precision_list.append(precision_score(y_true, y_pred, average='weighted'))
        recall_list.append(recall_score(y_true, y_pred, average='weighted'))
        auc_list.append(roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro'))

        # structure to build matrix
        y_true_matrix.extend(y_true)
        y_pred_matrix.extend(y_pred)

        # structure to build precision - recall
        y_pred_proba.extend(y_proba.max(axis=1))

        # structure to build loss vs epoch
        aux_history.append(history)

        n_class = 10

        # PRECISION VS RECALL PLOT per Fold
        precision = {}
        recall = {}
        for i in range(n_class):
            precision[i], recall[i], _ = precision_recall_curve(y_true, y_proba[:, i], pos_label=i)
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

        plt.title("Precision vs. Recall curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.savefig('Multiclass_PrecRecall_' + str(fold_count), dpi=300)
        plt.close()

        # ROC  - AUC PLOT per Fold
        fpr = {}
        tpr = {}
        thresh = {}

        for i in range(n_class):
            fpr[i], tpr[i], _ = roc_curve(y_true, y_proba[:, i], pos_label=i)
            plt.plot(fpr[i], tpr[i], label='Class {} vs Rest'.format(i))

        plt.title('Multiclass ROC curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend(loc='best')
        plt.savefig('Multiclass_ROC_' + str(fold_count), dpi=300)
        plt.close()

        tf.keras.backend.clear_session()

        fold_count += 1

    print('Acuraccy        : {:.8f} Std Dev: {:.3f}'.format(mean(accuracy), stdev(accuracy)))
    print('Precision       : {:.8f} Std Dev: {:.3f}'.format(mean(precision_list), stdev(precision_list)))
    print('Recall          : {:.8f} Std Dev: {:.3f}'.format(mean(recall_list), stdev(recall_list)))
    print('AUC             : {:.8f} Std Dev: {:.3f}'.format(mean(auc_list), stdev(auc_list)))

    # plot general confusion matrix over 10 ten folds
    cm = confusion_matrix(y_true_matrix, y_pred_matrix)
    plot_confusion_matrix(cm, LABELS, filename='general')
