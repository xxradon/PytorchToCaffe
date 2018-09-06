import numpy as np

# the blob with shape of (h,w,c) or (batch,h,w,c) for image
class Blob():
    def __init__(self,shape,father=None):
        shape=[int(i) for i in shape]
        self._data=None
        self.shape=[int(i) for i in list(shape)]
        self.father=type(father)==list and father or [father]

    @property
    def data(self):
        raise NotImplementedError('Blob.data is removed from this version of nn_tools, you should use .shape')

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def w(self):
        return self.shape[2]
    @property
    def h(self):
        return self.shape[1]

    @property
    def c(self):
        return self.shape[3]

    @property
    def batch_size(self):
        return self.shape[0]

    @property
    def dim(self):
        return len(self.shape)

    def new(self,father):
        return Blob(self.shape,father)
    def __getitem__(self, key):
        return self.shape[key]
    def __str__(self):
        return str(self.shape)
    def flaten(self):
        return Blob([np.prod(self.shape)])