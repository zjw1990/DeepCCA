import torch
import numpy as np 
def length_normalize(x):
    norms = np.sqrt(np.sum(x**2, axis=1))
    norms[norms == 0] = 1
    x /= norms[:, np.newaxis]
    return x

class cca_loss():
    def __init__(self, outdim_size, use_all_singular_values, device, r1, r2):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device
        self.r1 = r1
        self.r2 = r2
        # print(device)

    def loss(self, H1, H2):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        r1 = self.r1
        r2 = self.r2
        eps = 1e-9

        
        H1, H2 = H1.t(), H2.t()
        o1 = o2 = H1.size(0)
        #print(torch.isnan(H1).sum())
        #print(torch.isnan(H2).sum())
        
        assert torch.isnan(H1).sum().item() == 0
        assert torch.isnan(H2).sum().item() == 0
        

        o1 = o2 = H1.size(0)

        m = H1.size(1)
        #print(H1.size())

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)
        #H1Norm = H1bar/torch.norm(H1bar, dim=0)
        #H2Norm = H2bar/torch.norm(H2bar, dim=0)

        #print(torch.matrix_rank(H1Norm))
        #print(torch.matrix_rank(H2Norm))

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        
        #print(SigmaHat11)
        #print(SigmaHat12)
        #print(torch.matrix_rank(SigmaHat12))
        #print(torch.matrix_rank(SigmaHat11))
        #print(torch.matrix_rank(SigmaHat22))
        
        #print(torch.isnan(SigmaHat11).sum())
        #print(torch.isnan(SigmaHat12).sum())
        #print(torch.isnan(SigmaHat22).sum())

        #assert torch.isnan(SigmaHat11).sum().item() == 0
        #assert torch.isnan(SigmaHat12).sum().item() == 0
        #assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        #print('D1 is :', D1)
        #print('D2 is :', D2)
        #print(torch.isnan(D1).sum())
        #print(torch.isnan(D2).sum())
        #print(torch.isnan(V1).sum())
        #print(torch.isnan(V2).sum())

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        #print(posInd1.size())
        #print(posInd2.size())
        #print(torch.isnan(posInd1).sum())
        #print(torch.isnan(posInd2).sum())
        #print(torch.isnan(D1).sum())
        #print(torch.isnan(D2).sum())
        #print(torch.isnan(V1).sum())
        #print(torch.isnan(V2).sum())
        
        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
#         print(Tval.size())
        #print(torch.isnan(SigmaHat11RootInv).sum())
        #print(torch.isnan(SigmaHat22RootInv).sum())
        #print(torch.isnan(Tval).sum())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.trace(torch.matmul(Tval.t(), Tval))
            # print(tmp)
            corr = torch.sqrt(tmp)
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            sym = torch.matmul(Tval.t(), Tval)+r1 * torch.eye(o1, device=self.device)
            #print(torch.matrix_rank(sym))
            #print(torch.isnan(sym).sum())
            U, V = torch.symeig(sym, eigenvectors=True)
            #U = U[torch.gt(U, eps).nonzero()[:, 0]]
            U = U.topk(self.outdim_size)[0]
            #print('U is: ' ,U)
            corr = torch.sum(torch.sqrt(U))
            #print(torch.isnan(U).sum())
            #print(torch.isnan(V).sum())

        #print(corr)
        return -corr
