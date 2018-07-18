from matplotlib import pyplot
import arpys as arp
filename = ('/home/kevin/Documents/qmap/experiments/2018_07_CASSIOPEE/' + 
            'CaMnSb/S3_hv50_hv100_T230')
D = arp.dl.load_data(filename)

#    pyplot.ion()
figs = arp.pp.plot_cuts(D.data, dim=1)
print(len(figs))
pyplot.show()

