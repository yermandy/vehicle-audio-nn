import numpy as np
from tqdm import tqdm

####################################################################
# Probabilistic convolution/deconvolution for window size 2
####################################################################
class Events2:
    def __init__(self, n_events, seq_len):
        self.n_events = n_events
        self.seq_len = seq_len


    def conv(self, Px):
        Pc = np.zeros([self.n_events*2+1, self.seq_len-1])
        for i in range(self.seq_len-1):
            for c in range( self.n_events*2+1):
                for x1 in range( self.n_events+1):
                    for x2 in range(self.n_events+1):
                        if x1+x2==c:
                            Pc[c,i] += Px[x1,i]*Px[x2,i+1]

        return Pc

    def deconv( self, Pc, n_epochs=50 ):

        alpha = self.init_alpha()

        fit_quality = []
        for epoch in range(n_epochs):
            beta = self.get_beta( Pc, alpha )
            est_Px = self.get_px( beta )

            fit_quality.append( self.get_kl( Pc, est_Px))

            alpha = self.get_alpha( est_Px )

            
        est_Pc = np.zeros([self.n_events*2+1,self.seq_len-1])
        for i in range(self.seq_len-1):
            for c in range( self.n_events*2+1):
                nf = 0
                for x1 in range(self.n_events+1):
                    for x2 in range(self.n_events+1):
                        if x1+x2==c:
                            est_Pc[c,i] += est_Px[x1,i]*est_Px[x2,i]
            
        return est_Px, est_Pc, fit_quality


    # init alpha uniformly
    def init_alpha(self):
        alpha = np.zeros( [self.n_events+1,self.n_events+1,self.n_events*2+1,self.seq_len-1])

        for i in range(self.seq_len-1):
            for c in range( self.n_events*2+1):
                nf = 0
                for x1 in range(self.n_events+1):
                    for x2 in range(self.n_events+1):
                        if x1+x2==c:
                            alpha[x1,x2,c,i]=1
                            nf = nf + 1
                alpha[:,:,c,i] = alpha[:,:,c,i] / nf

        return alpha


    #
    def get_beta_pair( self, Pc, alpha ):

        beta1 = np.zeros(self.n_events+1 )
        beta2 = np.zeros(self.n_events+1 )

        for x1 in range(self.n_events+1):
            for c in range(x1,x1+self.n_events+1):
                beta1[x1] += Pc[c]*alpha[x1,c-x1,c]

        for x2 in range(self.n_events+1):
            for c in range(x2,x2+self.n_events+1):
                beta2[x2] += Pc[c]*alpha[c-x2,x2,c]

        return beta1, beta2
    

    #
    def get_beta(self, Pc, alpha):
        beta = np.zeros([self.n_events+1,self.seq_len])

        for i in range(self.seq_len-1):
            beta1, beta2 = self.get_beta_pair(Pc[:,i], alpha[:,:,:,i])

            beta[:,i] = beta[:,i] + beta1
            beta[:,i+1] = beta[:,i+1] + beta2

        return beta

    #
    def get_px(self, beta):
        est_Px = np.zeros([self.n_events+1,self.seq_len])
        for i in range(self.seq_len):
            est_Px[:,i] = beta[:,i]/np.sum( beta[:,i])

        return est_Px

    #
    def get_alpha(self, est_Px ):
        alpha = np.zeros( [self.n_events+1,self.n_events+1,self.n_events*2+1,self.seq_len-1])

        for i in range(self.seq_len-1):
            for c in range( self.n_events*2+1):
                nf = 0
                for x1 in range(self.n_events+1):
                    for x2 in range(self.n_events+1):
                        if x1+x2==c:
                            alpha[x1,x2,c,i]= est_Px[x1,i]*est_Px[x2,i+1]
                            nf = nf + alpha[x1,x2,c,i]
                alpha[:,:,c,i] = alpha[:,:,c,i] / nf

        return alpha

    #
    def get_kl( self, Pc, est_Px):
        F = 0
        for i in range(self.seq_len-1):
            for c in range( self.n_events*2+1):
                tmp = 0
                for x1 in range( self.n_events+1):
                    for x2 in range( self.n_events+1):
                        if x1+x2 == c:
                            tmp += est_Px[x1,i]*est_Px[x2,i+1]
                F += Pc[c,i]*np.log(tmp)

        return F

####################################################################
# Probabilistic deconvolution for generic window size
####################################################################
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
            

    def m_step( self ):
    
        self.Px.fill(0)
        for i in range( self.N-self.W+1):
            for w in range( self.W ):
                self.Px[:,i+w] += self.phi[i][:,:,w] @ self.alpha[:,i]

        self.Px = self.Px / np.sum( self.Px, axis=0 ) 

