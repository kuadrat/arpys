
from pyqtgraph import Qt as qt

class TracedVariable(qt.QtCore.QObject) :
    """ A pyqt implementaion of tkinter's/Tcl's traced variables using Qt's 
    signaling mechanism.
    Basically this is just a wrapper around any python object which ensures 
    that pyQt signals are emitted whenever the object is accessed or changed.

    In order to use pyqt's signals, this has to be a subclass of :class: 
    `QObject <pyqtgraph.Qt.QtCore.QObject>`.
    
    =============== ============================================================
    __value         the python object represented by this TracedVariable 
                    instance. Should never be accessed directly but only 
                    through the getter and setter methods.
    sigValueChanged :class: `Signal <pyqtgraph.Qt.QtCore.Signal>`; the signal 
                    that is emitted whenever :attr: self.__value is changed.
    sigValueRead    :class: `Signal <pyqtgraph.Qt.QtCore.Signal>`; the signal 
                    that is emitted whenever :attr: self.__value is read.
    =============== ============================================================
    """
    sigValueChanged = qt.QtCore.Signal()
    sigValueRead = qt.QtCore.Signal()

    def __init__(self, value=None) :
        # Have to call superclass init for signals to work
        super().__init__()
        self.__value = value

    def __repr__(self) :
        return '<TracedVariable({})>'.format(self.__value)

    def set_value(self, value=None) :
        """ Emit sigValueChanged and set the internal self.__value. """
        self.__value = value
        self.sigValueChanged.emit()

    def get_value(self) :
        """ Emit sigValueChanged and return the internal self.__value. 
        NOTE: the signal is emitted here before the caller actually receives 
        the return value. This could lead to unexpected behaviour. """
        self.sigValueRead.emit()
        return self.__value

    def on_change(self, callback) :
        """ Convenience wrapper for :class: `Signal 
        <pyqtgraph.Qt.QtCore.Signal>`'s 'connect'. 
        """
        self.sigValueChanged.connect(callback)

    def on_read(self, callback) :
        """ Convenience wrapper for :class: `Signal 
        <pyqtgraph.Qt.QtCore.Signal>`'s 'connect'. 
        """
        self.sigValueRead.connect(callback)

