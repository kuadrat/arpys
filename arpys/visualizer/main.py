import os

from pyqtgraph.Qt import QtGui, QtCore

from arpys.visualizer.detailsview import DetailsView

class Visualizer(QtGui.QMainWindow) :
    """ The main window of the arpys.visualizer application. Allows browsing 
    and inspecting files and opening them.
    """

    def __init__(self) :
        super().__init__()
        self.create_menu_bar()
        self.create_file_explorer()
        self.create_details_view()
        self.align()
        # FIXME This does not have an effect here, needs to be called later on.
        self.resize_columns()

        self.file_explorer.selectionModel().selectionChanged.connect(self.on_file_select)

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

    def create_details_view(self) :
        """ Create a View that displays all found metadata of selected file. """
        self.details_view = DetailsView()

    def align(self):
        """ Create window geometry. """
        self.resize(700, 600)

        self.central_widget = QtGui.QWidget()
        self.main_layout = QtGui.QGridLayout()
        self.central_widget.setLayout(self.main_layout)

        self.main_layout.setMenuBar(self.menu_bar)
        self.main_layout.addWidget(self.file_explorer, 0, 0)
        self.main_layout.addWidget(self.details_view, 0, 1)

        self.setCentralWidget(self.central_widget)

    def resize_columns(self) :
        """ Fit columns to content. """
        for i in self.shown_columns :
            self.file_explorer.resizeColumnToContents(i)

    def on_file_select(self) :
        """ Slot (=function to be executed) when user selects a new entry in 
        the TreeView.
        """
        # Get the path of selected file as a string
        index = self.file_explorer.currentIndex()
        path = self.file_explorer.model().filePath(index)
        print(path)
        self.details_view.update_details(path)

if __name__ == "__main__" :
    app = QtGui.QApplication([])

    v = Visualizer()
    v.show()
    v.resize_columns()

    app.exec_()
