from PyQt5 import QtWidgets
from QuestionPair import Ui_Dialog
from PyQt5.QtCore import pyqtSlot
from MaLSTM_GUI import modelPredict
import sys

class ApplicationWindow(QtWidgets.QDialog):
    def __init__(self):
        super(ApplicationWindow, self).__init__()

        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.setWindowTitle('QuestionPair PyQt5 GUI')
        self.ui.pushButton.clicked.connect(self.on_pushButton_clicked)
    @pyqtSlot()
    def on_pushButton_clicked(self):
        self.ui.lineEdit3.setText(str(modelPredict(self.ui.lineEdit1.text(),self.ui.lineEdit2.text())))

def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
 