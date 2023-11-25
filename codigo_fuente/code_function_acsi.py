import sys
import os
import zipfile
import shutil
import nibabel as nib
import tarfile
import os
from PIL import Image
import gzip
import dicom2nifti
import meshlib.mrmeshpy as mr
import meshlib.mrmeshnumpy as mrn
import cv2
import numpy as np
import scipy.signal
import torch
import monai
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QMainWindow
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtGui
from interfaz_acsi_choose import Ui_segunda_ventana
from interfaz_acsi_info import Ui_tercera_ventana
from PyQt5.uic import loadUi

from datetime import datetime
import tempfile
from glob import glob
from torch.utils.data import random_split, DataLoader
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns

# !pip install -q monai==1.1.0
# !pip install -q torch==1.10.2 torchtext torchvision
# !pip install -q torchio==0.18.73
# !pip install -q pytorch-lightning==1.5.10
# !pip install -q pandas==1.1.5 seaborn==0.11.1

# Cargar la interfaz de usuario desde el archivo .ui
qtCreatorFile = "interfaz_acsi.ui"
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
    
    def prepare_batch(self, batch):
        return batch['image'][tio.DATA], batch['label'][tio.DATA]
    
    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def forward(self, x):
        # Pasar la entrada a través de la red
        return self.net(x)
    
unet = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=3,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define el modelo (asegúrate de que sea la misma arquitectura que se usó para entrenar)
model = Model(
    net=unet,
    criterion=monai.losses.DiceCELoss(softmax=True),
    learning_rate=1e-2,
    optimizer_class=torch.optim.AdamW,
)

# Cargar los pesos del modelo
model_path = 'model_weights.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

# Poner el modelo en modo de evaluación
model.eval()


class MyApp(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.setupUi(self)
        self.setWindowTitle("Proyecto ACSI")
        self.logo_upch.setPixmap(QtGui.QPixmap("upch_logo.png"))
        self.logo_pucp.setPixmap(QtGui.QPixmap("pucp_logo.png"))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("upch_logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setFixedSize(705, 591)
        self.ventana_sec = Ventana_dos(parent=self)
        self.ventana_third = Ventana_tres(parent=self)
        # Conectar los botones a las funciones correspondientes
        self.load_file.activated.connect(self.agregar_carpeta)
        self.start_op.clicked.connect(self.inicio_tarea)
        self.save_file.clicked.connect(self.guardar_carpeta)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.play_next_image)
        
        self.next_both.clicked.connect(self.next_both_graphs)
        self.back_both.clicked.connect(self.back_both_graphs)
        self.stop_both.clicked.connect(self.stop_both_graphs)
        self.play_both.clicked.connect(self.play_both_graphs)
        self.choose.clicked.connect(self.choose_graph)
        self.button_info.clicked.connect(self.show_info)

    def show_info(self):
            try:
                self.ventana_third.show()    
            except Exception as e:
                print(f"Error: {e}")

    def choose_graph(self):
        try:
            self.ventana_sec.show()
        except Exception as e:
            print(f"Error: {e}")

    def recibir_new_frame(self, new_i_frame):
        try:
            self.index_orig=new_i_frame
            self.graph_1.setPixmap(QtGui.QPixmap(str(self.rutas_frame_orig[new_i_frame])))
            self.graph_2.setPixmap(QtGui.QPixmap(str(self.rutas_frame_seg[new_i_frame])))
            self.t_frame.setText(str(new_i_frame))
        except Exception as e:
            print(f"Error: {e}")
            
    def play_both_graphs(self):
        if not self.rutas_frame_orig == [] and not self.rutas_frame_seg == []:
            self.index_orig = 0
            self.index_orig = 0
            self.timer.start(60)  # Iniciar el QTimer para las imágenes originales

    def next_both_graphs(self):
        if not self.rutas_frame_orig == [] and not self.rutas_frame_seg == []:
            if self.index_orig < len(self.rutas_frame_orig) - 1:
                self.index_orig += 1
                self.graph_1.setPixmap(QtGui.QPixmap(str(self.rutas_frame_orig[self.index_orig])))
                self.t_frame.setText(str(self.index_orig))
            if self.index_orig < len(self.rutas_frame_seg) - 1:
                self.index_orig += 1
                self.graph_2.setPixmap(QtGui.QPixmap(str(self.rutas_frame_seg[self.index_orig])))
                self.t_frame.setText(str(self.index_orig))

    def back_both_graphs(self):
        if not self.rutas_frame_orig == [] and not self.rutas_frame_seg == []:
            if self.index_orig > 0:
                self.index_orig -= 1
                self.graph_1.setPixmap(QtGui.QPixmap(str(self.rutas_frame_orig[self.index_orig])))
                self.t_frame.setText(str(self.index_orig))
            if self.index_orig > 0:
                self.index_orig -= 1
                self.graph_2.setPixmap(QtGui.QPixmap(str(self.rutas_frame_seg[self.index_orig])))
                self.t_frame.setText(str(self.index_orig))


    def stop_both_graphs(self):
        try:
            self.timer.stop()  # Detener el QTimer
            self.t_frame.setText(str(self.index_orig))
        except Exception as e:
            print(f"Error: {e}")

    def play_next_image(self):
        try:
            # Actualizar frames de la imagen original
            if self.index_orig < len(self.rutas_frame_orig):
                self.graph_1.setPixmap(QtGui.QPixmap(str(self.rutas_frame_orig[self.index_orig])))
                self.index_orig += 1
            else:
                self.index_orig = 0  # Reiniciar el índice si ha llegado al final

            # Actualizar frames de la imagen segmentada
            if self.index_orig < len(self.rutas_frame_seg):
                self.graph_2.setPixmap(QtGui.QPixmap(str(self.rutas_frame_seg[self.index_orig])))
                self.index_orig += 1
            else:
                self.index_orig = 0  # Reiniciar el índice si ha llegado al final

            self.t_frame.setText(str(self.index_orig))  # Actualizar el texto del frame

        except Exception as e:
            print(f"Error: {e}")

            
    def agregar_carpeta(self):
        try:
            texto=self.load_file.currentText()
            escritorio = os.path.expanduser("~/Desktop")
            self.contenido_carpetas={}
            
            if texto == "Cargar NIfTI":
                archivo_nii_gz, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo NIfTI", escritorio, filter="Archivos NIfTI (*.nii.gz)")
                if not archivo_nii_gz:
                    return  # El usuario canceló la selección o no eligió un archivo NIfTI

                self.ruta_completa = archivo_nii_gz
                self.ruta_salida_nift = os.path.dirname(archivo_nii_gz)
                print(f"Archivo seleccionado: {archivo_nii_gz}")
                self.ruta_completa = archivo_nii_gz
                imagen_nii = nib.load(self.ruta_completa)
                self.datos = imagen_nii.get_fdata()
                print(self.datos.shape)
                self.flag_zip = 0
                self.mostrar_frames_graph_1(self.datos, self.flag_zip)

            # elif texto == "Cargar archivo ZIP":
            #         archivo_zip, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo ZIP", escritorio, filter="Archivos ZIP (*.zip)")
            #         self.flag_zip=1
            #         if not archivo_zip:
            #             return  # El usuario canceló la selección o no eligió un archivo ZIP
            #         # Abre el archivo ZIP en modo de lectura
            #         self.directorio_destino = os.path.dirname(archivo_zip)
            #         self.nombre_del_archivo = os.path.basename(archivo_zip)
            #         new_archivo_zip, extension = os.path.splitext(archivo_zip)
            #         if os.path.exists(new_archivo_zip):
            #             # Si la carpeta existe, elimínala junto con su contenido
            #             shutil.rmtree(new_archivo_zip)
            #         os.mkdir(new_archivo_zip)
            #         self.nombre_aux=os.path.basename(new_archivo_zip)
            #         print(new_archivo_zip)
            #         with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
            #             zip_ref.extractall(new_archivo_zip)
            #             print(f"Se seleccionó una carpeta: {new_archivo_zip}")
            #             lista_de_archivos = zip_ref.namelist()
            #             ruta_carpeta = os.path.join(new_archivo_zip, lista_de_archivos[0])
            #             ruta_carpeta = ruta_carpeta.replace('\\', '/')
            #             ruta_carpeta = ruta_carpeta[:-1]
            #             print(f"ruta:{ruta_carpeta}")
            #             self.nombre_aux=os.path.basename(ruta_carpeta)
            #             print(f"nombre_aux:{self.nombre_aux}")
            #             lista_de_archivos.pop(0)
            #             print(f"nombre: {lista_de_archivos}")
            #             self.ruta_salida_nift_1 = os.path.dirname(ruta_carpeta)
            #             print(f"ruta_salida_nift_1: {self.ruta_salida_nift_1}")     
            #             self.ruta_salida_nift = os.path.join(self.ruta_salida_nift_1, str(self.nombre_aux)+str("_nii_gz"))
            #             self.ruta_salida_nift=self.ruta_salida_nift.replace('\\', '/')
            #             print(self.ruta_salida_nift)
            #             os.makedirs(self.ruta_salida_nift, exist_ok=True)
            #             dicom2nifti.convert_directory(ruta_carpeta, self.ruta_salida_nift, compression=True, reorient=True)

            #             nombre_archivo_generado = os.listdir(self.ruta_salida_nift)[0]  # Suponiendo que solo haya un archivo
            #             # Renombra el archivo generado a tu nombre personalizado
            #             nombre_personalizado = str(self.nombre_aux)+str("_nii_gz.nii.gz")
            #             print(f"nombre_f:{nombre_personalizado}")
            #             nueva_ruta_archivo = os.path.join(self.ruta_salida_nift, nombre_personalizado)
            #             os.rename(os.path.join(self.ruta_salida_nift, nombre_archivo_generado), nueva_ruta_archivo)
  
            #             archivo_ni_gz = os.listdir(self.ruta_salida_nift)
            #             ruta_completa = os.path.join(self.ruta_salida_nift, archivo_ni_gz[0])
            #             self.ruta_completa = ruta_completa.replace('\\', '/')
            #             print(f"Ruta_nift:{self.ruta_completa}")
            #             # Cargar el archivo NIfTI
            #             imagen_nii = nib.load(self.ruta_completa)
            #             # Acceder a los datos de la imagen (esto puede variar según tus necesidades)
            #             self.datos = imagen_nii.get_fdata()
            #             print(self.datos.shape)
            #             self.mostrar_frames_graph_1(self.datos,self.flag_zip)

            elif texto == "Cargar DICOM":
                    archivo_tar, _ = QFileDialog.getOpenFileName(self, "Seleccionar Archivo TAR", escritorio, filter="Archivos TAR (*.tar.gz *.tar)")
                    self.flag_zip=1
                    if not archivo_tar:
                        return  # El usuario canceló la selección o no eligió un archivo ZIP
                    # Abre el archivo TAR en modo de lectura
                    self.directorio_destino = os.path.dirname(archivo_tar)
                    self.nombre_del_archivo = os.path.basename(archivo_tar)                    
                    archivo_tar, extension = os.path.splitext(archivo_tar)
                    if os.path.exists(archivo_tar):
                        # Si la carpeta existe, elimínala junto con su contenido
                        shutil.rmtree(archivo_tar)
                    os.mkdir(archivo_tar)
                    self.nombre_aux=os.path.basename(archivo_tar)
                    new_archivo_tar=os.path.join(self.directorio_destino, self.nombre_del_archivo)
                    new_archivo_tar = new_archivo_tar.replace('\\', '/')
                    
                    shutil.unpack_archive(new_archivo_tar, extract_dir=archivo_tar)
                    with tarfile.open(new_archivo_tar, 'r') as tar:
                        # Obtiene la lista de nombres de archivos en el archivo tar
                        lista_de_archivos = tar.getnames()
                    print(f"Se seleccionó una carpeta: {new_archivo_tar}")
                    ruta_carpeta = os.path.join(archivo_tar, lista_de_archivos[0])
                    ruta_carpeta = ruta_carpeta.replace('\\', '/')
                    print(f"Ruta: {ruta_carpeta}")
                    lista_de_archivos.pop(0)
                    print(f"nombre: {lista_de_archivos}")
                    self.nombre_aux=os.path.basename(ruta_carpeta)
                    self.ruta_salida_nift_1 = os.path.dirname(ruta_carpeta)
                    self.ruta_salida_nift = os.path.join(self.ruta_salida_nift_1, str(self.nombre_aux)+str("_nii_gz"))
                    self.ruta_salida_nift=self.ruta_salida_nift.replace('\\', '/')
                    print(self.ruta_salida_nift)
                    os.makedirs(self.ruta_salida_nift, exist_ok=True)
                    dicom2nifti.convert_directory(ruta_carpeta, self.ruta_salida_nift, compression=True, reorient=True)

                    nombre_archivo_generado = os.listdir(self.ruta_salida_nift)[0]  # Suponiendo que solo haya un archivo
                    # Renombra el archivo generado a tu nombre personalizado
                    nombre_personalizado = str(self.nombre_aux)+str("_nii_gz.nii.gz")
                    nueva_ruta_archivo = os.path.join(self.ruta_salida_nift, nombre_personalizado)
                    os.rename(os.path.join(self.ruta_salida_nift, nombre_archivo_generado), nueva_ruta_archivo)

                    archivo_ni_gz = os.listdir(self.ruta_salida_nift)
                    ruta_completa = os.path.join(self.ruta_salida_nift, archivo_ni_gz[0])
                    self.ruta_completa = ruta_completa.replace('\\', '/')
                    print(f"Ruta_nift:{self.ruta_completa}")
                    # Cargar el archivo NIfTI
                    imagen_nii = nib.load(self.ruta_completa)
                    # Acceder a los datos de la imagen (esto puede variar según tus necesidades)
                    self.datos = imagen_nii.get_fdata()
                    print(self.datos.shape)
                    self.mostrar_frames_graph_1(self.datos,self.flag_zip)
        except Exception as e:
            print(f"Error al mostrar las imágenes: {e}")   
            
    def mostrar_frames_graph_1(self, contenido_carpetas, flag_zip):
        try:
            ########Colocar el procesamiento para obtener frames en 2D########
            self.images_orig=contenido_carpetas
            # Crear una lista para almacenar las rutas
            self.rutas_frame_orig = []
            self.index_orig=0
            self.ruta_slices = os.path.join(os.path.dirname(self.ruta_salida_nift), "aux_slices")
            self.ruta_slices = self.ruta_slices.replace('\\', '/')
            print(f"dir_ruta_slices:{self.ruta_slices}")
            os.makedirs(self.ruta_slices, exist_ok=True)
            for i, imagen_2d in enumerate(contenido_carpetas[1,1,:]):
                # Convierte la imagen 2D en formato CV_8U (8 bits sin signo) para OpenCV
                imagen_2d = np.uint8(contenido_carpetas[:,:,i])
                # Define el nombre del archivo de imagen
                nombre_archivo = f"imagen_{i}.png"
                # Ruta completa del archivo de imagen
                ruta_archivo = os.path.join(self.ruta_slices, nombre_archivo)
                # Guarda la imagen como archivo
                cv2.imwrite(ruta_archivo, imagen_2d)
                # Agrega la ruta del archivo a la lista
                self.rutas_frame_orig.append(ruta_archivo)
                print(i)
            print(f"Se han guardado {len(self.rutas_frame_orig)} imágenes en {self.ruta_slices}")
            self.graph_1.setPixmap(QtGui.QPixmap(str(self.rutas_frame_orig[self.index_orig])))
            self.t_frame.setText(str(self.index_orig))
        except Exception as e:
            print(f"Error al mostrar las imágenes: {e}")

    def inicio_tarea(self):
        try:
            print("Iniciando segmentación...")
            self.segmentacion()
            self.mostrar_frames_graph_2()
        except Exception as e:
            print(f"Error en inicio_tarea: {e}")

    
    def get_preprocessing_transform(self):
        return tio.Compose([
            tio.RescaleIntensity((-1, 1)), # Reemplaza target_size con el tamaño deseado
            tio.EnsureShapeMultiple(8),   # Para U-Net
            tio.OneHot(),
        ])
    def convertir_a_stl(self, mask_array, predictions_folder):
        # Debes asegurarte de que 'mask_array' es una matriz 3D que representa tu segmentación
        # Si 'mask_array' no es 3D, necesitarás ajustarlo

        # Convertir la matriz 3D a SimpleVolume
        simpleVolume = mrn.simpleVolumeFrom3Darray(mask_array.astype(float))

        # Convertir SimpleVolume a FloatGrid
        floatGrid = mr.simpleVolumeToDenseGrid(simpleVolume)

        # Crear malla usando un iso-valor y el tamaño de voxel
        mesh = mr.gridToMesh(floatGrid, mr.Vector3f(0.1, 0.1, 0.1), 0.5)

        # Guardar la malla como STL
        stl_path = os.path.join(predictions_folder, 'mesh.stl')
        mr.saveMesh(mesh, stl_path)
        
    def segmentacion(self):
        try:
            subject = tio.Subject(image=tio.ScalarImage(self.ruta_completa))
            preprocess = self.get_preprocessing_transform()
            preprocessed_subject = preprocess(subject)

            predictions_folder = 'predictions'
            if not os.path.exists(predictions_folder):
                os.makedirs(predictions_folder)

            with torch.no_grad():
                input_tensor = preprocessed_subject['image'][tio.DATA][None]
                input_tensor = input_tensor.to(device)
                output = model(input_tensor)
                output_label = output.argmax(dim=1, keepdim=True).cpu()
                output_label_squeezed = output_label.squeeze(0)

                mask_array = output_label.numpy().squeeze()
                mask_array = (mask_array * 255).astype(np.uint8)

                # Guardar cada corte del volumen como PNG y la lista de rutas
                self.mascaras_obtenidas = []
                for i, slice in enumerate(mask_array):
                    slice_image = Image.fromarray(slice)
                    slice_path = os.path.join(predictions_folder, f'mask_slice_{i}.png')
                    slice_image.save(slice_path)
                    self.mascaras_obtenidas.append(slice_path)

                # Guardar la segmentación completa como archivo NIfTI
                nifti_path = os.path.join(predictions_folder, 'segmentacion_completa.nii.gz')
                segmented_image = tio.ScalarImage(tensor=output_label_squeezed)
                segmented_image.save(nifti_path)

                # Calcular el volumen
                # Asumiendo que cada voxel es 1x1x1 mm^3, pero verifica esto con tus datos
                volumen = np.sum(segmented_image.numpy() > 0)  # Cuenta los voxels en la ROI
                self.t_vol_h.setText(str(volumen)+"mm3")  # Muestra el volumen calculado
                self.convertir_a_stl(mask_array, predictions_folder)

            print("Segmentación completada y máscaras guardadas.")
        except Exception as e:
            print(f"Error en la segmentación: {e}")

            
    def mostrar_frames_graph_2(self):
        try:
            self.rutas_frame_seg = self.mascaras_obtenidas
            self.index_orig = 0

            if self.rutas_frame_seg:
                self.graph_2.setPixmap(QtGui.QPixmap(str(self.rutas_frame_seg[self.index_orig])))
                self.t_frame.setText(str(self.index_orig))
            else:
                print("No hay máscaras para mostrar.")
        except Exception as e:
            print(f"Error al mostrar las imágenes segmentadas: {e}")


    def guardar_carpeta(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        escritorio = os.path.expanduser("~/Desktop")
        carpeta_seleccionada = QFileDialog.getExistingDirectory(self, "Guardar Carpeta...", escritorio, options=options)

        if carpeta_seleccionada:
            try:
                carpeta_origen = self.ruta_nueva
                carpeta_destino = os.path.join(carpeta_seleccionada, self.nombre_aux+"_result")

                if os.path.exists(carpeta_destino):
                    contador = 1
                    while True:
                        nuevo_nombre = f"{self.nombre_aux} ({contador})"+"_result"
                        carpeta_destino = os.path.join(carpeta_seleccionada, nuevo_nombre)
                        if not os.path.exists(carpeta_destino):
                            break
                        contador += 1

                shutil.copytree(carpeta_origen, carpeta_destino)
                print("Carpeta de destino:", carpeta_destino)
                print("Guardado con éxito")
            except FileNotFoundError as e:
                print("Error: La carpeta de origen no se encuentra.")
            except PermissionError as e:
                print("Error: No tienes permisos para copiar la carpeta.")
            except Exception as e:
                print(f"Error al copiar la carpeta: {e}")

    def closeEvent(self, event):
        confirmacion = QtWidgets.QMessageBox.question(self, "Confirmar Salida", "¿Estás seguro de que deseas salir?",
                                                       QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if confirmacion == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

class Ventana_dos(QMainWindow):
    def __init__(self, parent=None):
        super(Ventana_dos, self).__init__(parent)
        loadUi("interfaz_acsi_choose.ui", self)
        self.acept_f.clicked.connect(self.close_segunda_ventana)
        self.setWindowTitle("Proyecto ACSI")
        self.setFixedSize(234, 84)

    def close_segunda_ventana(self):
        try:
            new_i_frame = int(self.c_frame.toPlainText())
            print("Valor de c_frame:", new_i_frame)
            self.parent().recibir_new_frame(new_i_frame)
            self.close()
        except Exception as e:
            print(f"Error: {e}")

class Ventana_tres(QMainWindow):
    def __init__(self, parent=None):
        super(Ventana_tres, self).__init__(parent)
        loadUi("interfaz_acsi_info.ui", self)
        self.setWindowTitle("Proyecto ACSI")
        self.setFixedSize(527, 306)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
