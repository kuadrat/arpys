import numpy as np
from data_slicer.utilities import TracedVariable

class DataHandler() :
    """ Base class for 2D and 3D data handlers. """

    def get_data(self) :
        """ Convenience `getter` method. Allows writing ``self.get_data()``
        instead of ``self.data.get_value()``. 
        """
        return self.data.get_value()

    def set_data(self, data=None, axes=None) :
        """ Convenience `setter` method. Allows writing ``self.set_data(d)`` 
        instead of ``self.data.set_value(d)``. 
        Additionally allows setting new *axes*. If *axes* is ``None``, the axes
        are reset to pixels.
        """
        if data is not None :
            self.data.set_value(data)
        if axes is not None :
            self.axes = axes
            self.main_window.set_axes()
        else :
            self.axes = EMPTY_AXES
        # Call on_z_dim_change here because it was not executed with the 
        # proper axes on the data change before
        self.on_z_dim_change()

    def get_main_data(self) :
        """ Return the 2d array that is currently displayed in the main plot. 
        """
        return self.main_window.main_plot.image_data

    def get_cut_data(self) :
        """ Return the 2d array that is currently displayed in the cut plot. 
        """
        return self.main_window.cut_plot.image_data

    def get_hprofile(self) :
        """ Return an array containing the y values displayed in the 
        horizontal profile plot (mw.y_plot).

        .. seealso::
            :func:`data_slicer.imageplot.CursorPlot.get_data`
        """
        return self.main_window.y_plot.get_data()[1]

    def get_vprofile(self) :
        """ Return an array containing the x values displayed in the 
        vertical profile plot (mw.x_plot).

        .. seealso::
            :func:`data_slicer.imageplot.CursorPlot.get_data`
        """
        return self.main_window.x_plot.get_data()[0]

    def get_iprofile(self) :
        """ Return an array containing the y values displayed in the 
        integrated intensity profile plot (mw.integrated_plot).

        .. seealso::
            :func:`data_slicer.imageplot.CursorPlot.get_data`
        """
        return self.main_window.integrated_plot.get_data()[1]

    def on_data_change(self) :
        """ Update self.main_window.image_data and replot. """
        logger.debug('on_data_change()')
        self.update_image_data()
        self.main_window.redraw_plots()
        # Also need to recalculate the intensity plot
        self.on_z_dim_change()

class DataHandler3D(DataHandler) :
    """ Object that keeps track of a set of 3D data and allows 
    manipulations on it. In a Model-View framework this could be seen as the 
    Model, while :class:`Viewer3D <arpys.visualizer.Viewer3D>` would be the 
    View part.
    """
    def __init__(self, main_window) :
        self.main_window = main_window

        # Initialize instance variables
        # np.array that contains the 3D data
        self.data = None
        self.axes = np.array([[0, 1], [0, 1], [0, 1]])
        # Indices of *data* that are displayed in the main plot 
        self.displayed_axes = (0,1)
        # Index along the z axis at which to produce a slice
        self.z = TracedVariable(0, name='z')
        ## Number of slices to integrate along z
        #integrate_z = TracedVariable(value=0, name='integrate_z')
        # How often we have rolled the axes from the original setup
        self._roll_state = 0

    def prepare_data(self, data, axes=3*[None]) :
        """ Load the specified data and prepare the corresponding z range. 
        Then display the newly loaded data.

        **Parameters**

        ====  ==================================================================
        data  3d array; the data to display
        axes  len(3) list or array of 1d-arrays or None; the units along the 
              x, y and z axes respectively. If any of those is *None*, pixels 
              are used.
        ====  ==================================================================
        """
        logger.debug('prepare_data()')

        self.data = TracedVariable(data, name='data')
        if axes is None :
            self.axes = np.array(3*[None])
        else :
            self.axes = np.array(axes)

        self.prepare_axes()
        self.on_z_dim_change()
        
        # Connect signal handling so changes in data are immediately reflected
        self.z.sig_value_changed.connect( \
            lambda : self.main_window.update_main_plot(emit=False))
        self.data.sig_value_changed.connect(self.on_data_change)

        self.main_window.update_main_plot()
        self.main_window.set_axes()

    def load(self, filename) :
        """ Alias to :func:`open <data_slicer.pit.PITDataHandler.open>`. """ 
        self.open(filename)

    def open(self, filename) :
        """ Open a file that's readable by :mod:`dataloading 
        <data_slicer.dataloading>`.
        """
        D = dl.load_data(filename)
        self.prepare_data(D.data, D.axes)

    def update_z_range(self) :
        """ When new data is loaded or the axes are rolled, the limits and 
        allowed values along the z dimension change.
        """
        # Determine the new ranges for z
        self.zmin = 0
        self.zmax = self.get_data().shape[2] - 1

        self.z.set_allowed_values(range(self.zmin, self.zmax+1))
#        self.z.set_value(self.zmin)

    def prepare_axes(self) :
        """ Create a list containing the three original x-, y- and z-axes 
        and replace *None* with the amount of pixels along the given axis.
        """
        shapes = self.data.get_value().shape
        # Avoid undefined axes scales and replace them with len(1) sequences
        for i,axis in enumerate(self.axes) :
            if axis is None :
                self.axes[i] = np.arange(shapes[i])

    def on_z_dim_change(self) :
        """ Called when either completely new data is loaded or the dimension 
        from which we look at the data changed (e.g. through :func:`roll_axes 
        <data_slicer.pit.PITDataHandler.roll_axes>`).
        Update the z range and the integrated intensity plot.
        """
        logger.debug('on_z_dim_change()')
        self.update_z_range()

        # Get a shorthand for the integrated intensity plot
        ip = self.main_window.integrated_plot
        # Remove the old integrated intensity curve
        try :
            old = ip.listDataItems()[0]
            ip.removeItem(old)
        except IndexError :
            pass

        # Calculate the integrated intensity and plot it
        self.calculate_integrated_intensity()
        ip.plot(self.integrated)

        # Also display the actual data values in the top axis
        zscale = self.axes[2]
        if zscale is not None :
            zmin = zscale[0]
            zmax = zscale[-1]
        else :
            zmin = 0
            zmax = self.data.get_value().shape[2]
        ip.set_secondary_axis(zmin, zmax)

    def calculate_integrated_intensity(self) :
        self.integrated = self.get_data().sum(0).sum(0)

    def update_image_data(self) :
        """ Get the right (possibly integrated) slice out of *self.data*, 
        apply postprocessings and store it in *self.image_data*. 
        Skip this if the z value happens to be out of range, which can happen 
        if the image data changes and the z scale hasn't been updated yet.
        """
        logger.debug('update_image_data()')
        z = self.z.get_value()
        integrate_z = \
        int(self.main_window.integrated_plot.slider_width.get_value()/2)
        data = self.get_data()
        try :
            self.main_window.image_data = make_slice(data, dim=2, index=z, 
                                                     integrate=integrate_z) 
        except IndexError :
            logger.debug(('update_image_data(): z index {} out of range for '
                          'data of length {}.').format(
                             z, self.image_data.shape[0]))

    def roll_axes(self, i=1) :
        """ Change the way we look at the data cube. While initially we see 
        an Y vs. X slice in the main plot, roll it to Z vs. Y. A second call 
        would roll it to X vs. Z and, finally, a third call brings us back to 
        the original situation.

        **Parameters**

        =  =====================================================================
        i  int; Number of dimensions to roll.
        =  =====================================================================
        """
        self._roll_axes(i, update=True)

    def _roll_axes(self, i=1, update=True) :
        """ Backend for :func:`roll_axes <arpys.pit.PITDataHandler.roll_axes>`
        that allows suppressing updating the roll-state.
        """
        logger.debug('roll_axes()')
        data = self.get_data()
        res = np.roll([0, 1, 2], i)
        self.axes = np.roll(self.axes, -i)
        self.set_data(np.moveaxis(data, [0, 1, 2], res), axes=self.axes)
        # Setting the data triggers a call to self.redraw_plots()
        self.on_z_dim_change()
        # Reset cut_plot's axes
        cp = self.main_window.cut_plot
        self.main_window.set_axes()
        if update :
            self._roll_state = (self._roll_state + i) % NDIM

class Foo() :
    """ The following functions really belong to the View-side, not the 
    Model-side. 
    """
    def lineplot(self, plot='main', dim=0, ax=None, n=10, offset=0.2, lw=0.5, 
                 color='k', label_fmt='{:.2f}', n_ticks=5, **getlines_kwargs) :
        """
        Create a matplotlib figure with *n* lines extracted out of one of the 
        visible plots. The lines are normalized to their global maximum and 
        shifted from each other by *offset*.
        See :func:`get_lines <data_slicer.utilities.get_lines>` for more 
        options on the extraction of the lines.
        This wraps the :class:`ImagePlot <data_slicer.imageplot.ImagePlot>`'s
        lineplot method.

        **Parameters**

        ===============  =======================================================
        plot             str; either "main" or "cut", specifies from which 
                         plot to extract the lines.
        dim              int; either 0 or 1, specifies in which direction to 
                         take the lines.
        ax               matplotlib.axes.Axes; the axes in which to plot. If 
                         *None*, create a new figure with a fresh axes.
        n                int; number of lines to extract.
        offset           float; spacing between neighboring lines.
        lw               float; linewidth of the plotted lines.
        color            any color argument understood by matplotlib; color 
                         of the plotted lines.
        label_fmt        str; a format string for the ticklabels.
        n_ticks          int; number of ticks to print.
        getlines_kwargs  other kwargs are passed to :func:`get_lines 
                         <data_slicer.utilities.get_lines>`
        ===============  =======================================================

        **Returns**

        ===========  ===========================================================
        lines2ds     list of Line2D objects; the drawn lines.
        xticks       list of float; locations of the 0 intensity value of 
                     each line.
        xtickvalues  list of float; if *momenta* were supplied, corresponding 
                     xtick values in units of *momenta*. Otherwise this is 
                     just a copy of *xticks*.
        xticklabels  list of str; *xtickvalues* formatted according to 
                     *label_fmt*.
        ===========  ===========================================================

        .. seealso::
            :func:`get_lines <data_slicer.utilities.get_lines>`
        """
        # Get the specified data
        if plot == 'main' :
            imageplot = self.main_window.main_plot
        elif plot == 'cut' :
            imageplot = self.main_window.cut_plot
        else :
            raise ValueError('*plot* should be one of ("main", "cut").')

        # Create a mpl axis object if none was given
        if ax is None : fig, ax = plt.subplots(1)

        return imageplot.lineplot(ax=ax, dim=dim, n=n, offset=offset, lw=lw, 
                                  color=color, label_fmt=label_fmt, 
                                  n_ticks=n_ticks, **getlines_kwargs)

    def plot_all_slices(self, dim=2, integrate=0, zs=None, labels='default', 
                        max_ppf=16, max_nfigs=2, **kwargs) :
        """ Wrapper for :func:`plot_cuts <data_slicer.utilities.plot_cuts>`.
        Plot all (or only the ones specified by `zs`) slices along dimension 
        `dim` on separate suplots onto matplotlib figures.

        **Parameters**

        =========  ============================================================
        dim        int; one of (0,1,2). Dimension along which to take the cuts.
        integrate  int or 'full'; number of slices to integrate around each 
                   extracted cut. If 'full', take the maximum number possible, 
                   depending on *zs* and whether the number of cuts is reduced 
                   due to otherwise exceeding *max_nfigs*.
        zs         1D np.array; selection of indices along dimension `dim`. 
                   Only the given indices will be plotted.
        labels     1D array/list of length z. Optional labels to assign to the 
                   different cuts. By default the values of the respective axis
                   are used. Set to *None* to suppress labels.
        max_ppf    int; maximum number of plots per figure.
        max_nfigs  int; maximum number of figures that are created. If more 
                   would be necessary to display all plots, a warning is 
                   issued and only every N'th plot is created, where N is 
                   chosen such that the whole 'range' of plots is represented 
                   on the figures. 
        kwargs     dict; keyword arguments passed on to :func:`pcolormesh 
                   <matplotlib.axes._subplots.AxesSubplot.pcolormesh>`. 
                   Additionally, the kwarg `gamma` for power-law color mapping 
                   is accepted.
        =========  ============================================================

        .. seealso::
            :func:`~data_slicer.utilities.plot_cuts`
        """
        data = self.get_data()
        if labels == 'default' :
            # Use the values of the respective axis as default labels
            labels = self.axes[dim]

        # The default values for the colormap are taken from the main_window 
        # settings
        gamma = self.main_window.gamma
        vmax = self.main_window.vmax * data.max()
        cmap = convert_ds_to_matplotlib(self.main_window.cmap, 
                                        self.main_window.cmap_name)
        plot_cuts(data, dim=dim, integrate=integrate, zs=zs, labels=labels, 
                  cmap=cmap, vmax=vmax, gamma=gamma, max_ppf=max_ppf, 
                  max_nfigs=max_nfigs)

    def overlay_model(self, model) :
        """ Display a model over the data. *model* should be function of two 
        variables, namely the currently displayed x- and y-axes.

        **Parameters**

        =====  =================================================================
        model  callable or :class:`Model <data_slicer.model.Model>`;
        =====  =================================================================

        .. seealso::
            :class:`Model <data_slicer.model.Model>`
        """
        if isinstance(model, FunctionType) :
            model = Model(model)
        elif not isinstance(model, Model) :
            raise ValueError('*model* has to be a function or a '
                             'data_slicer.Model instance')
        # Remove the old model
        self.remove_model()

        # Calculate model data in the required range and get an isocurve
        self.model = model
        # Bypass the minimum axes size limitation
        self.model.MIN_AXIS_LENGTH = 0
        model_axes = [self.axes[i] for i in self.displayed_axes]
        # Invert order for transposed view
        if self.main_window.main_plot.transposed.get_value() :
            self.model.set_axes(model_axes[::-1])
        else :
            self.model.set_axes(model_axes)
        self._update_isocurve()
        self._update_model_cut()

        # Connect signal handling
        self.z.sig_value_changed.connect(self._update_isocurve)
        self.main_window.cutline.sig_region_changed.connect(self._update_model_cut)

    def remove_model(self) :
        """ Remove the current model's visible and invisible parts. """
        # Remove the visible items from the plots
        try :
            self.main_window.main_plot.removeItem(self.iso)
            self.iso = None
            self.model = None
        except AttributeError :
            logger.debug('remove_model(): no model to remove found.')
            return
        # Remove signal handling
        try :
            self.z.sig_value_changed.disconnect(self._update_isocurve)
        except TypeError as e :
            logger.debug(e)
        try :
            self.main_window.cutline.sig_region_changed.disconnect(
                self._update_model_cut)
        except TypeError as e :
            logger.debug(e)

        # Redraw clean plots
        self.main_window.redraw_plots()

    def _update_isocurve(self) :
        try :
            self.iso = self.model.get_isocurve(self.z.get_value(), 
                                               axisOrder='row-major')
        except AttributeError :
            logger.debug('_update_isocurve(): no model found.')
            return
        # Make sure the isocurveItem is above the plot and add it to the main 
        # plot
        self.iso.setZValue(10)
        self.iso.setParentItem(self.main_window.main_plot.image_item)

    def _update_model_cut(self) :
        try :
            model_cut = self.main_window.cutline.get_array_region(
                            self.model.data.T,
                            self.main_window.main_plot.image_item,
                            self.displayed_axes)
        except AttributeError :
            logger.debug('_update_model_cut(): model or data not found.')
            return
        self.model_cut = self.main_window.cut_plot.plot(model_cut, 
                                                        pen=self.iso.pen)

