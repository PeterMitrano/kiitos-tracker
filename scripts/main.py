import sys
from PyQt5 import QtWidgets
from kiitos.kiitos_tracker import KiitosUi


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    _ = KiitosUi()
    app.exec()
