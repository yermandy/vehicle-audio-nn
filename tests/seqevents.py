import numpy as np

class Events:
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
