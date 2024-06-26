# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 19:52:48 2023

@author: OKTAY
"""

from PyQt5.QtWidgets import *
from arayuz import Ui_MainWindow
import os
import pandas as pd
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization
import seaborn as sns # visualization
import cv2
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import PIL
from PyQt5.QtGui import QPixmap
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
#from tensorflow.keras.applications.resnet50 import preprocess_input
#from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from PyQt5.QtCore import QThread, pyqtSignal
from keras.callbacks import Callback
from tensorflow.keras.models import load_model
from keras import backend as K
from keras.layers import Activation, Dense
from keras.applications.inception_v3 import InceptionV3, preprocess_input as preprocess_inception
from keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet50
from keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from keras.applications.vgg19 import VGG19, preprocess_input as preprocess_vgg19
from keras.applications.xception import Xception, preprocess_input as preprocess_xception
from keras.applications.mobilenet import MobileNet, preprocess_input as preprocess_mobilenet
from keras.applications.densenet import DenseNet121, preprocess_input as preprocess_densenet121
from keras.applications.densenet import DenseNet169, preprocess_input as preprocess_densenet169
from keras.applications.densenet import DenseNet201, preprocess_input as preprocess_densenet201
from keras.applications.nasnet import NASNetLarge, preprocess_input as preprocess_nasnetlarge
from keras.applications.nasnet import NASNetMobile, preprocess_input as preprocess_nasnetmobile
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    

class ModelPreprocessor:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def preprocess_input(self, x):
        if self.model_name == 'inception':
            return preprocess_inception(x)
        elif self.model_name == 'resnet50':
            return preprocess_resnet50(x)
        elif self.model_name == 'vgg16':
            return preprocess_vgg16(x)
        elif self.model_name == 'vgg19':
            return preprocess_vgg19(x)
        elif self.model_name == 'xception':
            return preprocess_xception(x)
        elif self.model_name == 'mobilenet':
            return preprocess_mobilenet(x)
        elif self.model_name == 'densenet121':
            return preprocess_densenet121(x)
        elif self.model_name == 'densenet169':
            return preprocess_densenet169(x)
        elif self.model_name == 'densenet201':
            return preprocess_densenet201(x)
        elif self.model_name == 'nasnetlarge':
            return preprocess_nasnetlarge(x)
        elif self.model_name == 'nasnetmobile':
            return preprocess_nasnetmobile(x)
        else:
            raise ValueError("Unsupported model name")
class WorkerThread(QThread):
    update_progress = pyqtSignal(int)
    stop_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.is_training = True
        self.stop_flag = False
       
    def run(self):
            while self.is_training:
                self.parent.arayuz.buttonModelEgit.setEnabled(False)
                self.model = self.choose_model()
                self.train_model(self.model)
                if self.stop_flag:
                    self.parent.arayuz.buttonModelEgit.setEnabled(True)
                    self.update_progress.emit(0)
                    self.stop_signal.emit()
                    break
                
                self.stop_training()
                self.parent.arayuz.buttonModelEgit.setEnabled(True)
                self.parent.arayuz.buttonDurdur.setEnabled(False)       
    def stop_training(self):
        self.is_training = False
        self.stop_flag = True
        
    def choose_model(self):
        if self.parent.arayuz.comboBox.currentText()=="Özel Model":
            model=self.create_model()
            print("özel")
        else:
            print("choose")
            self.parent.batch_size = int(self.parent.arayuz.editTextBatchSize.text() or 32)
            # Epoch al, boşsa 15 ata
            self.parent.epoch = int(self.parent.arayuz.editTextEpoch.text() or 15)
            model_path=self.parent.directory_path+'\\'+self.parent.selected_item
            model = load_model(model_path) 
            for layer in model.layers:
                layer.trainable = True

            #for layer in model.layers[-4:]:
            #    layer.trainable = True
            second_to_last_layer_output = model.layers[-2].output
            custom_output = Dense(len(self.parent.disease_types), activation='softmax',
                                  name='custom_output')(second_to_last_layer_output)
            model = Model(inputs=model.input, outputs=custom_output)
            
            model.summary()
            model.compile(optimizer='adam',
               loss=tf.keras.losses.categorical_crossentropy,
               metrics=['accuracy'])
        return model 
        
    
    def create_model(self): 
        # Batch Size al, boşsa 32 ata
        self.parent.batch_size = int(self.parent.arayuz.editTextBatchSize.text() or 32)
        # Epoch al, boşsa 15 ata
        self.parent.epoch = int(self.parent.arayuz.editTextEpoch.text() or 15)     
        if self.parent.selected_directory and self.parent.arayuz.comboBox.currentIndex() != -1:
            self.parent.labels=[key for key in self.parent.train_data.class_indices]
            self.parent.num_classes = len(self.parent.disease_types)
            
            self.model = keras.Sequential([
               layers.Rescaling(1./255, input_shape=(224,224, 3)),
                layers.Conv2D(16, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(32, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Conv2D(64, 3, padding='same', activation='relu'),
                layers.MaxPooling2D(),
                layers.Flatten(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dense(self.parent.num_classes,activation='softmax')
            ])
            self.model.compile(optimizer='adam',
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy'])
        return self.model

    def train_model(self, model):
        # Eğitim verilerini ve etiketlerini elde et        
        self.accuracy_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        
        self.model_callbacks = []
        progress_callback = TrainingProgressCallback(self)
        self.model_callbacks.append(progress_callback)
        if self.parent.arayuz.radioButtonCheckpoint.isChecked():
            model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min',verbose=2)
            self.model_callbacks.append(model_checkpoint)
            
        if self.parent.arayuz.radioButtonEarlyStopping.isChecked():
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=2, restore_best_weights=True)
            self.model_callbacks.append(early_stopping)
            
            print(self.model_callbacks)
            
        print(len(self.model_callbacks))
        if self.parent.arayuz.checkBoxKfold.isChecked():
            for i, fold in enumerate(self.parent.fold_data):
                
                self.parent.train_df=fold['train']
                test=fold['test']
                self.parent.valid_df, self.parent.test_df = train_test_split(test, test_size=0.5, random_state=23)
                self.parent.data_generator()
                self.history= self.model.fit(
                      self.parent.train_data,
                      validation_data=self.parent.valid_data,
                     #steps_per_epoch=len(train_data), bunu sor?
                      epochs=self.parent.epoch,
                      callbacks=self.model_callbacks
                      
                    )    
                # Test seti üzerinde tahmin yapın
                self.parent.labels=[key for key in self.parent.train_data.class_indices]
                y_test = self.parent.test_data.classes
                y_pred = self.model.predict(self.parent.test_data)
                y_pred = np.argmax(y_pred,axis=1)
            
                # Metrikleri hesaplayın ve listelere ekleyin
                accuracy = accuracy_score(y_test, y_pred)
                self.accuracy_scores.append(accuracy)
            
                precision = precision_score(y_test, y_pred,average='micro')
                self.precision_scores.append(precision)
            
                recall = recall_score(y_test, y_pred,average='micro')
                self.recall_scores.append(recall)
            
                f1 = f1_score(y_test, y_pred,average='micro')
                self.f1_scores.append(f1)
            
                
            for i in range(len(self.accuracy_scores)):
                print(f"\nMetrik Sonuçları - Fold {i + 1}")
                print(f"Accuracy: {self.accuracy_scores[i]}")
                print(f"Precision: {self.precision_scores[i]}")
                print(f"Recall: {self.recall_scores[i]}")
                print(f"F1 Score: {self.f1_scores[i]}")
           
                
               
        else:
            self.history= self.model.fit(
                  self.parent.train_data,
                  validation_data=self.parent.valid_data,
                 #steps_per_epoch=len(train_data), bunu sor?
                  epochs=self.parent.epoch,
                  callbacks=self.model_callbacks
                  
                )
        if self.parent.arayuz.radioButtonCheckpoint.isChecked():
            print("Model = Best Model")
            self.model=load_model("best_model.h5")
        

        
class TrainingProgressCallback(Callback):
    def __init__(self, worker_thread):
        super().__init__()
        self.worker_thread = worker_thread

    def on_epoch_end(self, epoch, logs=None):
        # Her epoch sonunda çağrılır ve progress bar'ı günceller
        progress_percentage = int((epoch + 1) / self.worker_thread.parent.epoch * 100)
        self.worker_thread.update_progress.emit(progress_percentage)
        if not self.worker_thread.is_training:
           self.model.stop_training = True
        
            
class main(QMainWindow):
    def __init__(self)-> None:
        super().__init__()
        self.arayuz= Ui_MainWindow()
        self.arayuz.setupUi(self)
        
        
        self.arayuz.buttonDurdur.setEnabled(False)
        # Python dosyalarının bulunduğu dizini belirtin
        self.arayuz.comboBox.addItem("Özel Model")
       
        self.load_model_directory=""
        current_directory = os.getcwd()
        model_folder_name = "model"
        # Model klasörünün tam yolunu oluştur
        model_folder_path = os.path.join(current_directory, model_folder_name)
        print(model_folder_path)
        self.directory_path = model_folder_path
        self.selected_item=""
        self.selected_directory = None
        self.batch_size = 32
        print()
        # Epoch al, boşsa 15 ata
        self.epoch = 2
        self.directorys=[]
        self.data_control=False
        self.gercek_deger=""
        self.test_data = None 
        self.train_data = None
        self.valid_data = None
        self.df=None
        
        # ComboBox'ı doldur
        self.populate_combobox()
        self.onComboBoxPreprocessChanged()
        self.worker_thread = WorkerThread(self)
        self.worker_thread.update_progress.connect(self.update_progress_bar)

        # ComboBox'ın seçimini izle
        self.arayuz.comboBox.currentIndexChanged.connect(self.handle_combobox_selection)
        
        # Butona tıklandığında dosya seçtirme penceresini aç
        self.arayuz.buttonVeriSeti.clicked.connect(self.open_file_dialog)
        
        # Tabloya tıklandığında çağrılacak işlevi bağla
        self.arayuz.tableTrain.itemClicked.connect(self.cell_clicked)
        self.arayuz.tableTest.itemClicked.connect(self.cell_clicked)
        self.arayuz.tableValidation.itemClicked.connect(self.cell_clicked)
        self.arayuz.buttonModelEgit.clicked.connect(self.model_egit)
        self.arayuz.buttonDurdur.clicked.connect(self.stop_training)
        self.worker_thread.stop_signal.connect(self.on_stop_signal)
        self.arayuz.buttonSonucAl.clicked.connect(self.show_results)
        self.arayuz.buttonTahmin.clicked.connect(self.model_predict)
        self.arayuz.buttonModelEkle.clicked.connect(self.open_model_dialog)
        self.arayuz.comboBoxModel.currentIndexChanged.connect(self.onComboBoxIndexChanged)
        self.arayuz.comboBox_2.currentIndexChanged.connect(self.onComboBoxPreprocessChanged)
        self.arayuz.buttonVeriSec.clicked.connect(self.open_file)
        self.arayuz.buttonKaydet.clicked.connect(self.save_model)
        self.arayuz.editTextBatchSize.editingFinished.connect(self.on_editing_finished)
       
        self.arayuz.checkBoxKfold.stateChanged.connect(self.checkboxStateChanged)
        self.arayuz.comboBoxKfold.currentIndexChanged.connect(self.changeComboBoxKfold)
        
    def changeComboBoxKfold(self):
        index=self.arayuz.comboBoxKfold.currentIndex()
        print(len(self.fold_data))
        print(index)
        fold=self.fold_data[index]
        #self.valid_df=fold['valid']
        self.train_df=fold['train']
        test=fold['test']
        self.valid_df, self.test_df = train_test_split(test, test_size=0.5, random_state=23)
        self.data_generator()
        self.populate_table(self.arayuz.tableTrain,self.train_df)
        self.populate_table(self.arayuz.tableTest,self.test_df)
        self.populate_table(self.arayuz.tableValidation,self.valid_df)
        
        
        
    def checkboxStateChanged(self,state):
        if state == 2:
            self.arayuz.comboBoxKfold.clear()
            self.kfold_split(self.df)
        if state == 0:
            self.arayuz.comboBoxKfold.clear()
            
            
            
        
    def on_editing_finished(self):
        
        self.batch_size=int(self.arayuz.editTextBatchSize.text())
        self.data_generator()
        
        
        
    def save_model(self):
        save_path = self.get_model_save_path()
        
        if save_path:
            # Modeli belirtil qen yere kaydet
            self.worker_thread.model.save(save_path+".h5")
            print(f"Model başarıyla kaydedildi: {save_path}")
        else:
            print("Kayıt işlemi iptal edildi.")
        
        
    def onComboBoxPreprocessChanged(self):
  
        self.model_name=self.arayuz.comboBox_2.currentText()
        print(self.model_name)
        self.preprocessor = ModelPreprocessor(self.model_name)
        if self.test_data or self.train_data or self.valid_data:
            self.data_generator()
        
   
        
        
        
    def model_predict(self):
        from tensorflow.keras.preprocessing import image
        
        all_predictions = []
        labels=[key for key in self.train_data.class_indices]
        print("LABELSSS:",labels)
        image_path = self.arayuz.buttonVeriSec.text()
        self.arayuz.label
        img = image.load_img(image_path, target_size=(224, 224))
       
        # Görüntüyü diziye çevir
        img_array = image.img_to_array(img)
        
        # Giriş boyutuna uygun hale getir
        img_array = np.expand_dims(img_array, axis=0)
        
        img_array = self.preprocessor.preprocess_input(img_array)
       

        if self.arayuz.radioButtonSecilen.isChecked():
            model = load_model(self.load_model_directory)
            input_shape = model.input_shape

            # Giriş şeklini yazdır
            print("Input Shape:", input_shape)
            
            # Image width ve height değerlerini al
            image_width = input_shape[1]
            image_height = input_shape[2]
            img = image.load_img(image_path, target_size=(image_width, image_height))
           
            # Görüntüyü diziye çevir
            img_array = image.img_to_array(img)
            
            # Giriş boyutuna uygun hale getir
            img_array = np.expand_dims(img_array, axis=0)
            img_array = self.preprocessor.preprocess_input(img_array)
            predictions = model.predict(img_array)
            # Tahmin yap
            predict = model.predict(img_array)
            y_pred = np.argmax(predict, axis=1)
            for i in range(len(y_pred)):
                highest_probability = np.max(predict[i])
                tahmin = labels[y_pred[i]]
                print(f'Modelin Tahmini : {tahmin}, Olasılık: {highest_probability * 100:.2f}%')
            print(y_pred)
            self.arayuz.labelTahmin.setText(f'Modelin Tahmini : {tahmin}, Olasılık: {highest_probability * 100:.2f}%')
            if self.gercek_deger == tahmin:
                self.arayuz.labelTahmin.setStyleSheet("background-color: rgba(0, 255, 0, 100);")
            elif self.gercek_deger=="": 
                self.arayuz.labelTahmin.setStyleSheet("background-color: rgba(0, 0, 255, 100);")
            else:
                self.arayuz.labelTahmin.setStyleSheet("background-color: rgba(255, 0, 0, 100);")
                    
        elif self.arayuz.radioButtonEgitilen.isChecked():
            model = self.worker_thread.model
            input_shape = model.input_shape

            # Giriş şeklini yazdır
            print("Input Shape:", input_shape)
            
            # Image width ve height değerlerini al
            image_width = input_shape[1]
            image_height = input_shape[2]
            img = image.load_img(image_path, target_size=(image_width, image_height))
           
            # Görüntüyü diziye çevir
            img_array = image.img_to_array(img)
            
            # Giriş boyutuna uygun hale getir
            img_array = np.expand_dims(img_array, axis=0)
            img_array = self.preprocessor.preprocess_input(img_array)
            predictions = model.predict(img_array)
            # Tahmin yap
            predict = model.predict(img_array)
            print(predict)
            y_pred = np.argmax(predict, axis=1)
            for i in range(len(y_pred)):
               highest_probability = np.max(predict[i])
               tahmin = labels[y_pred[i]]
               print(f'Modelin Tahmini : {tahmin}, Olasılık: {highest_probability * 100:.2f}%')
               print(y_pred)
               self.arayuz.labelTahmin.setText(f'Modelin Tahmini : {tahmin}, Olasılık: {highest_probability * 100:.2f}%')
            if self.gercek_deger == tahmin:
                self.arayuz.labelTahmin.setStyleSheet("background-color: rgba(0, 255, 0, 100);")
            elif self.gercek_deger=="": 
                self.arayuz.labelTahmin.setStyleSheet("background-color: rgba(0, 0, 255, 100);")
            else:
                self.arayuz.labelTahmin.setStyleSheet("background-color: rgba(255, 0, 0, 100);")

        elif self.arayuz.radioButtonTumu.isChecked():
            print("GİRDİ")
            for file_path in self.directorys:
                model = load_model(file_path)
                input_shape = model.input_shape
                # Giriş şeklini yazdır
                print("Input Shape:", input_shape)
                # Image width ve height değerlerini al
                image_width = input_shape[1]
                image_height = input_shape[2]
                img = image.load_img(image_path, target_size=(image_width, image_height))
                # Görüntüyü diziye çevir
                img_array = image.img_to_array(img) 
                # Giriş boyutuna uygun hale getir
                img_array = np.expand_dims(img_array, axis=0)
                img_array = self.preprocessor.preprocess_input(img_array)
                predictions = model.predict(img_array)
                print(predictions)
                all_predictions.append(predictions)
            if all_predictions:
                average_predictions = np.mean(all_predictions, axis=0)
                print(average_predictions)
                y_pred = np.argmax(average_predictions, axis=1)
                print(y_pred)
                for i in range(len(y_pred)):
                  highest_probability = np.max(average_predictions[i])
                  tahmin = labels[y_pred[i]]
                  print(f'Modelin Tahmini : {tahmin}, Olasılık: {highest_probability * 100:.2f}%')
                  self.arayuz.labelTahmin.setText(f'Modelin Tahmini : {tahmin}, Olasılık: {highest_probability * 100:.2f}%')
                if self.gercek_deger == tahmin:
                    self.arayuz.labelTahmin.setStyleSheet("background-color: rgba(0, 255, 0, 100);")
                elif self.gercek_deger=="": 
                    self.arayuz.labelTahmin.setStyleSheet("background-color: rgba(0, 0, 255, 100);")
                else:
                    self.arayuz.labelTahmin.setStyleSheet("background-color: rgba(255, 0, 0, 100);")


                

                
            
            
        

    def onComboBoxIndexChanged(self, index):
        if 0 <= index < len(self.directorys):
            self.load_model_directory = self.directorys[index]
            print(f"Seçilen Dizin: {self.load_model_directory}")
            print(index)
        else:
            print("Geçersiz index",index)
            
    def closeEvent(self, event):
        # Pencere kapatıldığında iş parçacığını sonlandır
        self.worker_thread.stop_training()

        # İş parçacığını terminate et
        self.worker_thread.terminate()
        event.accept()    
        
    def show_results(self):
        
        self.labels=[key for key in self.train_data.class_indices]
        y_test = self.test_data.classes
        
        if self.arayuz.radioButtonSecilen.isChecked():
            model = load_model(self.load_model_directory)
        elif self.arayuz.radioButtonEgitilen.isChecked():
            model = self.worker_thread.model
            
        if not self.arayuz.radioButtonTumu.isChecked():
            y_pred = model.predict(self.test_data)
            y_pred = np.argmax(y_pred,axis=1)
        else:
            all_predictions = []
            for file_path in self.directorys:
                model = load_model(file_path)
                predictions = model.predict(self.test_data)
                print(predictions)
                all_predictions.append(predictions)
            if all_predictions:
                average_predictions = np.mean(all_predictions, axis=0)
                print(average_predictions)
                y_pred = np.argmax(average_predictions, axis=1)
                
            
        if self.arayuz.checkBoxKfold.isChecked():
            average_accuracy = np.mean(self.worker_thread.accuracy_scores)
            average_precision = np.mean(self.worker_thread.precision_scores)
            average_recall = np.mean(self.worker_thread.recall_scores)
            average_f1 = np.mean(self.worker_thread.f1_scores)
            self.arayuz.labelSonuc.append(f'Ortalama Accuracy: {average_accuracy}')
            self.arayuz.labelSonuc.append(f'Ortalama Precision: {average_precision}')
            self.arayuz.labelSonuc.append(f'Ortalama Recall: {average_recall}')
            self.arayuz.labelSonuc.append(f'Ortalama F1 Score: {average_f1}')
            
            self.arayuz.labelSonuc_2.append(f'Ortalama Accuracy: {average_accuracy}')
            self.arayuz.labelSonuc_2.append(f'Ortalama Precision: {average_precision}')
            self.arayuz.labelSonuc_2.append(f'Ortalama Recall: {average_recall}')
            self.arayuz.labelSonuc_2.append(f'Ortalama F1 Score: {average_f1}')
                
        else:
            report=classification_report(y_test,y_pred,target_names = self.labels)
            
            # Metni HTML biçimine çevir
            html_report_text = f"<pre style='font-size: 13pt;'>{report}</pre>"


            self.arayuz.labelSonuc.setHtml(html_report_text)
            self.arayuz.labelSonuc_2.setHtml(html_report_text)
            print(report)
        from sklearn.metrics import ConfusionMatrixDisplay
       
        
        conf_mat = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=self.labels)
        cmap=sns.color_palette("rocket", as_cmap=True)
        disp.plot(cmap=cmap)
        plt.xticks(rotation=25)
        # tight_layout fonksiyonunu kullanarak içeriği düzenle
        plt.tight_layout()
        plt.savefig('confusion_matrix.png',dpi=300)
        pixmap1 = QPixmap('confusion_matrix.png')
        self.arayuz.labelMatris.setPixmap(pixmap1)
        self.arayuz.labelMatris.setScaledContents(True)
        self.arayuz.labelMatris_2.setPixmap(pixmap1)
        self.arayuz.labelMatris_2.setScaledContents(True)
        plt.close()  
        
        plt.plot(range(1, len(self.worker_thread.history.history['val_accuracy']) + 1),self.worker_thread.history.history['accuracy'])
        plt.plot(range(1, len(self.worker_thread.history.history['val_accuracy']) + 1),self.worker_thread.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Val'])
        plt.tight_layout()
        plt.savefig('accuracy.png',dpi=300)
        pixmap2 = QPixmap('accuracy.png')
        self.arayuz.labelAcc.setPixmap(pixmap2)
        self.arayuz.labelAcc.setScaledContents(True)
        plt.close()
        
        plt.plot(range(1, len(self.worker_thread.history.history['loss']) + 1),self.worker_thread.history.history['loss'])
        plt.plot(range(1, len(self.worker_thread.history.history['loss']) + 1),self.worker_thread.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Val'])
        plt.tight_layout()
        plt.savefig('loss.png',dpi=300)
        pixmap3 = QPixmap('loss.png')
        self.arayuz.labelLoss.setPixmap(pixmap3)
        self.arayuz.labelLoss.setScaledContents(True)
        plt.close()
                
        
    def model_egit(self):
        if self.selected_directory:
            
            self.arayuz.progressBar.setValue(0)
            self.arayuz.buttonDurdur.setEnabled(True)
            self.worker_thread.is_training = True
            self.worker_thread.stop_flag = False
            self.worker_thread.start()
                
    def stop_training(self):
        # İş parçacığını durdu
        self.arayuz.buttonDurdur.setEnabled(False)
        self.worker_thread.stop_training()
        self.arayuz.buttonModelEgit.setText("Eğitim Durduruluyor...")

    def update_progress_bar(self, value):
        self.arayuz.progressBar.setValue(value)
        
    def on_stop_signal(self):
        self.arayuz.buttonModelEgit.setText("MODELİ EĞİT")
                
                     
        
    def cell_clicked(self, item):
        row = item.row()
        col = item.column()  
        value = item.text()
        print(f'Tıklanan Hücre: Satır={row}, Sütun={col}, Değer={value}')
        if col==1:
            self.arayuz.buttonVeriSec.setText(value)
            self.show_image(value)
            self.gercek_deger=self.sender().item(row, 0).text()
            self.arayuz.labelGercekDeger.setText("Gerçek Değer: "+self.gercek_deger)
            self.arayuz.labelTahmin.setText("Modelin Tahmini:")
            self.arayuz.labelTahmin.setStyleSheet("")
                     
                    
        
    def show_image(self, path):
        pixmap = QPixmap(path)
        self.arayuz.labelGoruntu.setPixmap(pixmap)
          
        
    def populate_combobox(self):
        # Dizin içindeki .h5 uzantılı dosyaları al
        python_files = [file for file in os.listdir(self.directory_path) if file.endswith(".h5")]

        # ComboBox'ı dosya isimleriyle doldur
        self.arayuz.comboBox.addItems(python_files)
        self.arayuz.comboBox.setCurrentIndex(0)
        self.selected_item = self.arayuz.comboBox.currentText()
    
    def handle_combobox_selection(self):
        # ComboBox'tan seçilen değeri al
        self.selected_item = self.arayuz.comboBox.currentText()
        # Seçilen dosyanın adını göster
        #self.arayuz.label.setText(f"Seçilen Dosya: {self.selected_item}")
    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly  # Sadece okuma izni
        file_name, _ = QFileDialog.getOpenFileName(self, "Dosya Seç", "", "Resim Dosyaları (*.png *.jpg *.jpeg);;Tüm Dosyalar (*)", options=options)
        if file_name:
            self.gercek_deger=""
            self.arayuz.labelGercekDeger.setText("Gerçek Değer:")
            self.arayuz.buttonVeriSec.setText(file_name)
            self.show_image(file_name)

    def open_file_dialog(self):     
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly

        directory_dialog = QFileDialog()
        directory_dialog.setOptions(options)
        directory_dialog.setFileMode(QFileDialog.Directory)  # Sadece bir dizin seçme modu
        # Kullanıcıdan dizin seçmesini iste
        self.selected_directory = directory_dialog.getExistingDirectory(self, "Dizin Seç", 
         "",options=QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        
        if self.selected_directory:
            # Butonun metnini güncelle
            self.arayuz.buttonVeriSeti.setText(self.selected_directory)
            self.df=self.populate_dataframe()
         
            self.split_data(self.df)
            self.populate_table(self.arayuz.tableTrain,self.train_df)
            self.populate_table(self.arayuz.tableTest,self.test_df)
            self.populate_table(self.arayuz.tableValidation,self.valid_df)
                
    def open_model_dialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Dosya Seç', '/home', 'H5 Files (*.h5);;All Files (*)') 
        if fname[0]:  # Eğer dosya seçildiyse
            # Dosya yolunu sakla
            self.directorys.append(fname[0])
            file_name = fname[0].split("/")[-1]  # Dosya adını al
            self.arayuz.comboBoxModel.addItem(file_name)  # ComboBox'a ekle
            self.arayuz.comboBoxModel.setCurrentIndex(self.arayuz.comboBoxModel.findText(file_name))  # Son seçileni göster
           
            print(self.directorys)

            
            
    def populate_dataframe(self):
        data_path = self.selected_directory
        # Veri setindeki klasör isimlerini alın
        self.disease_types = [folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))]
        disease_types=self.disease_types
        print(disease_types)
        # DataFrame için bir liste oluşturun
        from tqdm import tqdm  # Süreci göstermek için kullanılıyor
        df_list = []
        
        # Her bir hastalık türü için görüntü dosyalarının yollarını içeren DataFrame oluşturun
        for disease_type in disease_types:
            disease_folder_path = os.path.join(data_path, disease_type)
            for image_name in tqdm(os.listdir(disease_folder_path), desc= str(disease_type)):
                image_path = os.path.join(disease_folder_path, image_name)
                # disease_type_numeric yaparak gelen string değerleri indekse göre numaralandırarak yazıyorum
                # disease_type direkt yazdırılabilir ve sayı olarak değil isim olarak ifade edilebilir.
                ##disease_type_numeric = disease_types.index(disease_type)
                df_list.append({'disease_type': disease_type,'image': image_path})
        # DataFrame'i oluşturun
        df = pd.DataFrame(df_list)
        return df
    
    def get_model_save_path(self):
       options = QFileDialog.Options()
       options |= QFileDialog.DontUseNativeDialog  # Windows'ta native dosya seçim penceresini kullanmayı devre dışı bırak

       file_path, _ = QFileDialog.getSaveFileName(self, "Modeli Kaydet", "", "H5 Model Dosyaları (*.h5);;Tüm Dosyalar (*)", options=options)

       return file_path

            
            
    def split_data(self,df):
        size=self.arayuz.spinBox.value()/100   
        # Veriyi eğitim ve geriye kalan olarak ayırır
        self.train_df, remaining_df = train_test_split(df, test_size=size, random_state=6)

        # Kalan veriyi doğrulama ve test alt kümelerine ayırır
        self.valid_df, self.test_df = train_test_split(remaining_df, test_size=0.5, random_state=23)
        self.data_generator()
    
    def kfold_split(self,df):
        from sklearn.model_selection import KFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        # K-fold çapraz doğrulama için n_splits belirtin
        kfold = KFold(n_splits=3, shuffle=True,random_state=42)
        # Her bir fold için eğitim ve test verilerini saklamak için boş listeler oluşturduk
        self.fold_data = []
        i=1
        # K-fold çapraz doğrulama için döngü
        for train_index, test_index in kfold.split(df):
            train, test = df.iloc[train_index], df.iloc[test_index]
            # Her bir fold'un verilerini bir sözlük içinde sakla
            self.fold_data.append({'train': train,'test': test,})
            self.arayuz.comboBoxKfold.addItem("Fold "+str(i))
            self.arayuz.comboBoxKfold.setCurrentIndex(0)
            i=i+1
            
          
        
    def data_gen(self,train,test):
        
        self.valid, self.test = train_test_split(test, test_size=0.5, random_state=23)
        datagen = ImageDataGenerator(preprocessing_function=self.preprocessor.preprocess_input)
        if self.arayuz.checkBox.isChecked():
                print("ÇOKLAMA YAPILDI")
                train_datagen = ImageDataGenerator(
                    rotation_range=20, width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,
                    zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
                
                train_data = train_datagen.flow_from_dataframe(
                    dataframe=test,
                    x_col='image',
                    y_col='disease_type',
                    target_size=(224, 224),
                    class_mode='categorical',
                    batch_size=self.batch_size,
                    shuffle=True,
                    # save_to_dir=artirma_dizini,  # Artırılmış görüntüleri kaydetmek için bir dizin belirtin
                    # save_prefix='aug',
                    # save_format='jpeg'
                )
        else:   
                print("Çoklama yapılmadı")
                # Train veri setini kullanarak bir veri akışı oluşturun
                train_data = datagen.flow_from_dataframe(
                    dataframe=train,
                    x_col='image',
                    y_col='disease_type',
                    target_size=(224, 224),
                    class_mode='categorical',
                    batch_size=self.batch_size,
                    shuffle=True,
                    # save_to_dir=artirma_dizini,  # Artırılmış görüntüleri kaydetmek için bir dizin belirtin
                    # save_prefix='aug',
                    # save_format='jpeg'
                )
                
                
        # Test veri setini kullanarak bir veri akışı oluşturun
        test_data = datagen.flow_from_dataframe(
            dataframe=test,
            x_col='image',
            y_col='disease_type',
            target_size=(224, 224),
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False  # Test setinde karıştırma yapma
        )
        # Doğrulama veri setini kullanarak bir veri akışı oluşturun
        valid_data = datagen.flow_from_dataframe(
              dataframe=valid,
              x_col='image',
              y_col='disease_type',
              target_size=(224, 224),
              class_mode='categorical',
              batch_size=self.batch_size,
              shuffle=False,
              #subset='validation'
          )
        return train_data,test_data,valid_data
        
    def data_generator(self):
        datagen = ImageDataGenerator(preprocessing_function=self.preprocessor.preprocess_input)
        if self.arayuz.checkBox.isChecked():
                print("ÇOKLAMA YAPILDI")
                train_datagen = ImageDataGenerator(
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    #horizontal_flip=True,
                    fill_mode='nearest')
                self.train_data = train_datagen.flow_from_dataframe(
                    dataframe=self.train_df,
                    x_col='image',
                    y_col='disease_type',
                    target_size=(224, 224),
                    class_mode='categorical',
                    batch_size=self.batch_size,
                    shuffle=True,
                    # save_to_dir=artirma_dizini,  # Artırılmış görüntüleri kaydetmek için bir dizin belirtin
                    # save_prefix='aug',
                    # save_format='jpeg'
                )
        else:   
                print("Çoklama yapılmadı")
                # Train veri setini kullanarak bir veri akışı oluşturun
                self.train_data = datagen.flow_from_dataframe(
                    dataframe=self.train_df,
                    x_col='image',
                    y_col='disease_type',
                    target_size=(224, 224),
                    class_mode='categorical',
                    batch_size=self.batch_size,
                    shuffle=True,
                    # save_to_dir=artirma_dizini,  # Artırılmış görüntüleri kaydetmek için bir dizin belirtin
                    # save_prefix='aug',
                    # save_format='jpeg'
                )
        # Doğrulama veri setini kullanarak bir veri akışı oluşturun
        self.valid_data = datagen.flow_from_dataframe(
              dataframe=self.valid_df,
              x_col='image',
              y_col='disease_type',
              target_size=(224, 224),
              class_mode='categorical',
              batch_size=self.batch_size,
              shuffle=False,
              #subset='validation'
          )
        
        # Test veri setini kullanarak bir veri akışı oluşturun
        self.test_data = datagen.flow_from_dataframe(
            dataframe=self.test_df,
            x_col='image',
            y_col='disease_type',
            target_size=(224, 224),
            class_mode='categorical',
            batch_size=self.batch_size,
            shuffle=False  # Test setinde karıştırma yapma
        )
                
          
    def populate_table(self,table,df):
        
        # DataFrame'in sütun ve satırlarını döngü ile tarayın
        table.setRowCount(df.shape[0])
        table.setColumnCount(df.shape[1])
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iat[row, col]))  # DataFrame hücresini alın ve QTableWidgetItem oluşturun
                table.setItem(row, col, item)  # QTableWidgetItem'i tabloya ekleyin

app=QApplication([])
window=main()
window.show()
app.exec_()
