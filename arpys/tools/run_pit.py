#!/usr/bin/python

import argparse
import sys

import pkg_resources
from pyqtgraph.Qt import QtGui

from arpys.pit import mainwindow as mw

def run_pit(args=sys.argv) :
    parser = argparse.ArgumentParser()
    # The first argument of sys.argv is the path to the executable
    parser.add_argument('executable', nargs=1)
    parser.add_argument('filename', nargs='?', default='')

    parsed = parser.parse_args(args)

    app = QtGui.QApplication([])

    main_window = mw.MainWindow()
    if parsed.filename :
        filename = parsed.filename
    else :
        path = pkg_resources.resource_filename('arpys', 'pit/')
        filename = path + 'example_data.p'

    main_window.data_handler.prepare_data(filename)
    app.exec_()

