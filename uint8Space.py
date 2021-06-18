import numpy as np
from gym.spaces import Space

class uint8Space(Space):
    
    """
    Example::

        >>> uint8Space((2,5))
        >>> uint8Space(7)

    """

    def __init__(self, shape = None):
        assert (shape is not None), 'Safety: a binaryArray must have some shape'
        if np.isscalar(shape):
            self.shape = (shape,)
        else:
            assert (len(shape) <= 2)
            self.shape = shape
        super(uint8Space, self).__init__(self.shape, np.uint8)


    def sample(self):
        return np.uint8(self.np_random.randint(0, 256, self.shape))
            

    def contains(self, x):
        result = False
        if type(x) == np.ndarray:
            shapeOfX = x.shape
            if (shapeOfX == self.shape):
                if (x.dtype == 'uint8'):
                    result = True
        return result
    
    def __repr__(self):
        result =  "uint8Array(" + str(self.shape) + ")"
        return result

    def __eq__(self, other):
        # Omer Sella: there is a redundancy here: if the dimensions are the same then the shape must be as well, but for completeness I'm leaving it in.
        return ( isinstance(other, uint8Space) and (self.shape == other.shape) )


