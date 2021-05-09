from lattice import lattice

import numpy as np

class fftgrid:

    def __init__(self, latt : lattice, ecutrho : np.float64):
    
        self.ns = np.array([0.0, 0.0, 0.0])

        gmax = np.sqrt(2.0 * ecutrho)
        
        for i in range(3):
            vlen = np.linalg.norm(latt.get_vector(i))
            self.ns[i] = np.ceil( 2.0 * gmax * vlen / 2.0 / np.pi )

    def get_size(self):
        return self.ns

    def get_ntot(self):
        return self.ns[0] * self.ns[1] * self.ns[2]

    def get_n1(self):
        return self.ns[0]

    def get_n2(self):
        return self.ns[1]

    def get_n3(self):
        return self.ns[2]

    def get_n1_left(self):
        return fft_left_end(self.ns[0])

    def get_n1_right(self):
        return fft_right_end(self.ns[0])

    def get_n2_left(self):
        return fft_left_end(self.ns[1])

    def get_n2_right(self):
        return fft_right_end(self.ns[1])
    
    def get_n3_left(self):
        return fft_left_end(self.ns[2])

    def get_n3_right(self):
        return fft_right_end(self.ns[2])

        
def fft_left_end(n) -> int:
        
    if n % 2 == 0:
        return -(n - 2) / 2
    else:
        return -(n - 1) / 2

def fft_right_end(n) -> int:

    if n % 2 == 0:
        return n /2
    else:
        return (n - 1) / 2

if __name__ == '__main__':
    a = [ 8.0, 0.0, 0.0 ]
    b = [ 0.0, 8.0, 0.0 ]
    c = [ 0.0, 0.0, 12.0 ]
    
    latt = lattice(a, b, c)

    fftgrid = fftgrid(latt, 10.0)
    
    print(fftgrid.ns)
