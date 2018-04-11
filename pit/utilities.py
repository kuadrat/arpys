
import numpy as np
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
    sigAllowedValuesChanged
                    :class: `Signal <pyqtgraph.Qt.QtCore.Signal>`; the signal 
                    that is emitted whenever :attr: self.allowed_values are set
                    or unset.
    allowed_values  :class: `array <numpy.ndarray>`; a sorted list of all values
                    that self.__value can assume. If set, all tries to set the 
                    value will automatically set it to the closest allowed one.
    =============== ============================================================
    """
    sigValueChanged = qt.QtCore.Signal()
    sigValueRead = qt.QtCore.Signal()
    sigAllowedValuesChanged = qt.QtCore.Signal()
    allowed_values = None

    def __init__(self, value=None) :
        # Have to call superclass init for signals to work
        super().__init__()
        self.__value = value

    def __repr__(self) :
        return '<TracedVariable({})>'.format(self.__value)

    def set_value(self, value=None) :
        """ Emit sigValueChanged and set the internal self.__value. """
        # Choose the closest allowed value
        if self.allowed_values is not None :
            value = self.find_closest_allowed(value)
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

    def set_allowed_values(self, values=None) :
        """ Define a set/range/list of values that are allowed for this 
        Variable. Once set, all future calls to set_value will automatically 
        try to pick the most reasonable of the allowed values to assign. 
        Emits :signal: `sigAllowedValueChanged`

        ====== =================================================================
        values iterable; The complete list of allowed (numerical) values. This
               is converted to a sorted np.array internally. If values is 
               `None`, all restrictions on allowed values will be lifted and 
               all values are allowed.
        ====== =================================================================
        """
        if values is None :
            # Reset the allowed values, i.e. all values are allowed
            self.allowed_values = None
            self.min_allowed = None
            self.max_allowed = None
        else :
            # Convert to sorted numpy array
            try :
                values = np.array(values)
            except TypeError :
                message = 'Could not convert allowed values to np.array.'
                raise TypeError(message)

            # Sort the array for easier indexing later on
            values.sort()
            self.allowed_values = values

            # Store the max and min allowed values (necessary?)
            self.min_allowed = values[0]
            self.max_allowed = values[-1]

        self.sigAllowedValuesChanged.emit()

    def find_closest_allowed(self, value) :
        """ Return the value of the element in self.allowed_values (if set) 
        that is closest to `value`. 
        """
        if self.allowed_values is None :
            return value
        else :
            ind = np.abs( self.allowed_values-value ).argmin()
            return self.allowed_values[ind]


