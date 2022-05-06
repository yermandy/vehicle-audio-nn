####################################################################
# Probabilistic deconvolution for generic window size
####################################################################
# %%
import numpy as np
from tqdm import tqdm
import numba

class Events:
    
    def __init__(self, W ):
        self.W = W

    def deconv( self, Pc, n_epochs = 50 ):

        self.K = Pc.shape[0]-1
        
        if self.K % self.W != 0:
            raise "The number of rows minus one must be divisible by window size."

        self.Kx = int( self.K / self.W )
        
        self.dims = tuple( [self.Kx+1 for i in range(self.W)] )

        self.N = Pc.shape[1]+self.W-1        
        self.alpha = np.zeros( [(self.Kx+1)**self.W, self.N-self.W+1] )

        # random init 
        self.Px = np.random.rand( self.Kx+1,self.N)
        self.Px = self.Px / np.sum( self.Px, axis=0)

        self.Pc = np.zeros( Pc.shape )

        #
        self.phi = []
        for i in range( self.N-self.W+1 ):
            phii = np.zeros( (self.Kx+1,(self.Kx+1)**self.W, self.W) )
            for w in range( self.W ):
                for z in range( (self.Kx+1)**self.W ):
                    ind = np.unravel_index( z, self.dims )
                    phii[ind[w],z,w] += Pc[np.sum(ind),i]

            self.phi.append( phii )
        
        # run EM
        self.conv( )
        obj = [ np.sum( Pc * np.log( self.Pc ) ) ]

        for i in tqdm( range( n_epochs )):
            self.e_step()
            self.m_step()

            self.conv( )
            obj.append( np.sum( Pc * np.log( self.Pc ) ))

        self.obj = np.array( obj )

        return np.copy( self.Px), np.copy( self.Pc), np.copy( self.obj )



    def conv( self, Px=None ):

        if Px is not None:
            self.N = Px.shape[1]
            self.Kx = Px.shape[0]-1
            self.K = self.Kx*self.W
            self.Px = np.copy( Px )
            self.dims = tuple( [self.Kx+1 for i in range(self.W)] )
            self.Pc = np.zeros( [self.K + 1, self.N-self.W+1] )

        self.Pc.fill( 0 )
        for i in range( self.N-self.W+1):
            for z in range( (self.Kx+1)**self.W ):
                ind = np.unravel_index( z, self.dims )

                tmp = 1
                for j in range( self.W ):
                    tmp = tmp*self.Px[ind[j],i+j]

                self.Pc[ np.sum(ind),i ] += tmp

        if Px is not None:
            return np.copy( self.Pc )

    def e_step( self ):

        norm_const = np.zeros( self.W*(self.Kx+1) )

        for i in range( self.N-self.W+1):
            norm_const.fill(0)
            for z in range( (self.Kx+1)**self.W ):
                ind = np.unravel_index( z, self.dims )

                tmp = 1
                for j in range( self.W ):
                    tmp = tmp*self.Px[ind[j],i+j]

                self.alpha[z,i] = tmp
                norm_const[ np.sum( ind ) ] += tmp

            for z in range( (self.Kx+1)**self.W ):
                ind = np.unravel_index( z, self.dims )
                self.alpha[z,i] = self.alpha[z,i] / norm_const[ np.sum(ind) ]
    
    @numba.jit(nopython=True)
    def m_step( self ):
    
        self.Px.fill(0)
        for i in range( self.N-self.W+1):
            for w in range( self.W ):
                self.Px[:,i+w] += self.phi[i][:,:,w] @ self.alpha[:,i]

        self.Px = self.Px / np.sum( self.Px, axis=0 ) 

# %%
if __name__ == "__main__":
    Kx = 5   # max num events per dense window; x1 is from {0,1,...,Kx}
    N  = 100  # length of the dense sequence x1,...xN
    W  = 2   # window size

    Px = np.random.rand(Kx + 1, N)
    Px /= Px.sum(0)

    A = Events(W)
    Pc = A.conv(Px)

    Px_est, Pc_est, kl_hist = A.deconv(Pc, 200)


# %%
