# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'interfaz_acsi_choose.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_segunda_ventana(object):
    def setupUi(self, segunda_ventana):
        segunda_ventana.setObjectName("segunda_ventana")
        segunda_ventana.resize(236, 88)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("upch_logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        segunda_ventana.setWindowIcon(icon)
        segunda_ventana.setStyleSheet("background-color: rgb(76, 76, 76);")
        self.centralwidget = QtWidgets.QWidget(segunda_ventana)
        self.centralwidget.setObjectName("centralwidget")
        self.c_frame = QtWidgets.QTextEdit(self.centralwidget)
        self.c_frame.setGeometry(QtCore.QRect(100, 20, 41, 31))
        self.c_frame.setStyleSheet("QTextEdit{\n"
"color: rgb(6, 5, 50);\n"
"font: 9pt \"Segoe MDL2 Assets\";\n"
"background-color:rgb(167, 190, 203);\n"
"border-style:outset;\n"
"border-top-left-radius: 4px;\n"
"border-radius: 4px;\n"
"border-width: 2px;\n"
"padding: 2px;\n"
"border-color: rgb(34, 48, 38);\n"
"}")
        self.c_frame.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.c_frame.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.c_frame.setObjectName("c_frame")
        self.l_frame_1 = QtWidgets.QLabel(self.centralwidget)
        self.l_frame_1.setGeometry(QtCore.QRect(10, 20, 81, 31))
        self.l_frame_1.setStyleSheet("QLabel{\n"
"color: rgb(6, 5, 50);\n"
"font: 8pt \"Segoe MDL2 Assets\";\n"
"background-color:rgb(167, 190, 203);\n"
"border-style:outset;\n"
"border-radius: 4px;\n"
"border-width: 1px;\n"
"padding: 2px;\n"
"border-color: rgb(34, 48, 38);\n"
"}")
        self.l_frame_1.setObjectName("l_frame_1")
        self.acept_f = QtWidgets.QPushButton(self.centralwidget)
        self.acept_f.setGeometry(QtCore.QRect(150, 20, 61, 31))
        self.acept_f.setStyleSheet("QPushButton{\n"
"color: rgb(6, 5, 50);\n"
"font: 9pt \"Segoe MDL2 Assets\";\n"
"background-color:rgb(221, 230, 237);\n"
"border-style:outset;\n"
"border-top-left-radius: 4px;\n"
"border-radius: 4px;\n"
"border-width: 1.5px;\n"
"padding: 2px;\n"
"border-color: rgb(27, 38, 30);\n"
"}")
        self.acept_f.setObjectName("acept_f")
        segunda_ventana.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(segunda_ventana)
        self.statusbar.setObjectName("statusbar")
        segunda_ventana.setStatusBar(self.statusbar)

        self.retranslateUi(segunda_ventana)
        QtCore.QMetaObject.connectSlotsByName(segunda_ventana)

    def retranslateUi(self, segunda_ventana):
        _translate = QtCore.QCoreApplication.translate
        segunda_ventana.setWindowTitle(_translate("segunda_ventana", "MainWindow"))
        self.l_frame_1.setText(_translate("segunda_ventana", "Escoger frame:"))
        self.acept_f.setText(_translate("segunda_ventana", "Aceptar"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    segunda_ventana = QtWidgets.QMainWindow()
    ui = Ui_segunda_ventana()
    ui.setupUi(segunda_ventana)
    segunda_ventana.show()
    sys.exit(app.exec_())
