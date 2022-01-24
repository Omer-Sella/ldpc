# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:21:27 2020

@author: Omer
"""
# Omer Sella: A binary array is a space that weas not implemented in openAI gym, 
# so I added its definition here. The reason I need to define it as a space is 
# to comply with the openAI gym implementation, that states the observation space 
# and action space.



import numpy as np
from gym.spaces import Space

class binarySpace(Space):
    r"""A binary array in :math:`\{ 0, 1, \\dots, n-1 \}`. 

    Example::

        >>> binarySpace((2,5))
        >>> binarySpace(7)

    """

    def __init__(self, shape = None):
        #assert (shape is not None), 'Safety: a binaryArray must have some shape'
        #self.shape = shape
        #if np.isscalar(shape):
        #    self.shape = (shape,)
        #else:
        #    assert (len(shape) <= 2)
        #    self.shape = shape
        #OSS diabeling space extension
        super(binarySpace, self).__init__((shape,), np.int64)


    def sample(self):
        return self.np_random.randint(0, 2, self.shape)
            

    def contains(self, x):
        result = False
        if type(x) == np.ndarray:
            shapeOfX = x.shape
            if (shapeOfX == self.shape):
                if ((x == 1) | (x == 0)).all():
                    result = True
        return result
    
    def __repr__(self):
        result =  "binaryArray(" + str(self.shape) + ")"
        return result

    def __eq__(self, other):
        # Omer Sella: there is a redundancy here: if the dimensions are the same then the shape must be as well, but for completeness I'm leaving it in.
        return ( isinstance(other, binarySpace) and (self.shape == other.shape) )
