import os

from pyqtgraph.Qt import QtGui, QtCore

class Visualizer(QtGui.QMainWindow) :
    """ The main window of the arpys.visualizer application. Allows browsing 
    and inspecting files and opening them.
    """

    def __init__(self) :
        super().__init__()
        self.create_menu_bar()
        self.create_file_explorer()
        self.align()
        # FIXME This does not have an effect here, needs to be called later on.
        self.resize_columns()

    def create_menu_bar(self):
        """ Create the menu entries. """
        menu_bar = QtGui.QMenuBar()
        # status_bar = QStatusBar()

        file_menu = menu_bar.addMenu('&File')
        open_dir = QtGui.QAction('About', self)
        open_dir.setStatusTip('Show information about this app')
        open_dir.triggered.connect(lambda : print('This is a work in progress.'))
        file_menu.addAction(open_dir)
        file_menu.addSeparator()
        self.menu_bar = menu_bar

    def create_file_explorer(self):
        """ Create or update the file explorer view (QTreeView). """
        model = QtGui.QFileSystemModel()
        file_explorer = QtGui.QTreeView()
        file_explorer.setModel(model)

        # Only allow browsing directories below "Home"
        home = QtCore.QDir.homePath()
        model.setRootPath(home)
        file_explorer.setRootIndex(model.index(home))

        # Select the current working directory
        cwd = QtCore.QDir.currentPath()
        file_explorer.setCurrentIndex(model.index(cwd))

        # Visual fiddling: the columns are 0: filename, 1: size, 2: date 
        # modified, 3: type
        self.shown_columns = [0, 1, 2, 3]
        self.hidden_columns = [2, 3]
        # Remove unnecessary columns
        for i in self.hidden_columns :
            file_explorer.hideColumn(i)
            self.shown_columns.remove(i)

        self.file_explorer = file_explorer

    def align(self):
        """ Create window geometry. """
        self.resize(700, 600)

        self.central_widget = QtGui.QWidget()
        self.main_layout = QtGui.QGridLayout()
        self.central_widget.setLayout(self.main_layout)

        self.main_layout.setMenuBar(self.menu_bar)
        self.main_layout.addWidget(self.file_explorer, 0, 0)
#        self.main_layout.addWidget(self.details_panel, 0, 1)

        self.setCentralWidget(self.central_widget)

    def resize_columns(self) :
        """ Fit columns to content. """
        for i in self.shown_columns :
            self.file_explorer.resizeColumnToContents(i)

    def set_directory(self, path) :
        pass

if __name__ == "__main__" :
    app = QtGui.QApplication([])

    v = Visualizer()
    v.show()
    v.resize_columns()

    app.exec_()
