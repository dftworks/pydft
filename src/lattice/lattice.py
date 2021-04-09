import numpy as np

import numpy.typing as npt

class lattice:

    def __init__(self, a : npt.ArrayLike, b , c):
        self.data = np.zeros( (3,3), order = 'F' )
        self.data[:,0] = a
        self.data[:,1] = b
        self.data[:,2] = c

    def vector_a(self):
        return self.data[:,0]

    def vector_b(self):
        return self.data[:,1]

    def vector_c(self):
        return self.data[:,2]

    def volume(self):
        return np.dot( self.data[:,0], np.cross(self.data[:,1], self.data[:,2]) )

    def get_vector(self, idx):
        return self.data[:,idx]
    
    def reciprocal_lattice(self):
        a = self.data[:,0]
        b = self.data[:,1]
        c = self.data[:,2]

        factor = 2.0 * np.pi / self.volume()
        
        return lattice( np.cross(b,c) * factor, \
                        np.cross(c,a) * factor, \
                        np.cross(a,b) * factor )

    def __str__(self):
        des = ""
        for i in range(3):
            des = des + \
                  "%16.12f\t%16.12f\t%16.12f\n" % \
                  ( self.data[0,i], self.data[1,i], self.data[2,i] )
        return des

if __name__ == '__main__':
    a = [ 8.0, 0.0, 0.0 ]
    b = [ 0.0, 8.0, 0.0 ]
    c = [ 0.0, 0.0, 12.0 ]
    
    latt = lattice(a, b, c)
    
    print( latt )

    blatt = latt.reciprocal_lattice()

    print( blatt )
    
    for i in range(3):
        for j in range(3):
            print( "%16.12f" % \
                   np.dot(latt.get_vector(i), blatt.get_vector(j)), end = "\t" )
        print()
