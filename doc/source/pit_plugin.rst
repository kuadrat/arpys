PIT plugin
==========

The ``arpys`` module has been developed in conjunction with the Python Image 
Tool (PIT), provided by the `data_slicer 
module <https://github.com/kuadrat/data_slicer>`_.
To use them in together, the ``ds_arpes_plugin`` is required.
This is a separate python module, which can also be installe dvia ``pip``::

   pip install ds_arpes_plugin

See the `documentation of the data_slicer package 
<https://data-slicer.readthedocs.io/en/latest/plugins.html#plugins>`_ for 
details on how plugins work in PIT.

The ``ds_arpes_plugin`` mostly wraps some of arpys' functionality, most 
prominently the loading of ARPES data.
But it also provides a few  additional features. To get a full overview, 
visit its own documentation.
