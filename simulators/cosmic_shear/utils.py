# Copyright (c) 2014-2015, CosmicPy Developers
# Licensed under CeCILL 2.1 - see LICENSE.rst
'''
    :mod:`cosmicpy.utils` -- Utility functions
    =========================================
    .. module:: cosmicpy.utils
    :synopsis: Utility functions
    .. moduleauthor:: Francois Lanusse <francois.lanusse@cea.fr>
    .. moduleauthor:: Anais Rassat <anais.rassat@epfl.ch>
    .. Created on Jun 17, 2013 by Francois Lanusse
    '''


def z2a(z):
    """ converts from redshift to scale factor """
    return 1.0/(1.0 + z)


def a2z(a):
    """ converts from scale factor to  redshift """
    return 1.0/a - 1.0
