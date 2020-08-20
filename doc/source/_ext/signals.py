from docutils import nodes

def signal_role(name, rawtext, lineno, inliner, options={}, content=[]) :
    """ Don't actually do anything, but having this suppresses warnings when 
    the :signal: role is used.
    """
    return [nodes.reference(rawtext, '', refuri='')], []

def setup(app) :
    app.add_role('signal', signal_role)
    return
