import numpy as np
import torch

class Propagator():
    def __init__(self,opt):
        self.E = opt.energy
        self.pxs = opt.pxs
        self.z = opt.z
        self.wavelength = 12.4 / self.E * 1e-10
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fresnel_prop(self, Di):
        nz,n1,n2,layer = [torch.tensor(i,device=self.device) for i in Di.shape]
        pi,lamda,z,pxs = [torch.tensor(i,device=self.device) for i in [np.pi,self.wavelength,self.z,self.pxs]]
        fx = (torch.cat((torch.arange(0,n2/2),torch.arange(-n2/2,0)))).to(self.device)/pxs/n2
        fy = (torch.cat((torch.arange(0,n1/2),torch.arange(-n1/2,0)))).to(self.device)/pxs/n1
        fx,fy = torch.meshgrid(fx,fy)
        f2 = fx**2 + fy**2
        angle = -pi*lamda*z*f2
        H_real = torch.cos(angle)
        H_imag = torch.sin(angle)
        D1 = torch.fft(self.batch_ifftshift2d(Di),signal_ndim=2)
        D1_real,D1_imag = torch.unbind(D1,-1)
        DH_real =D1_real*H_real - D1_imag*H_imag
        DH_imag = D1_real*H_imag + D1_imag*H_real
        DH = torch.stack((DH_real,DH_imag),-1)
        Do = self.batch_fftshift2d(torch.ifft(DH,signal_ndim=2))
        Do_real,Do_imag = torch.unbind(Do,-1)
        Idet = Do_real**2 + Do_imag**2
        return Idet

    def roll_n(self,X, axis, n):
        f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
        b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
        front = X[f_idx]
        back = X[b_idx]
        return torch.cat([back, front], axis)

    def batch_fftshift2d(self,x):
        # Provided by PyTorchSteerablePyramid
        real, imag = torch.unbind(x, -1)
        for dim in range(1, len(real.size())):
            n_shift = real.size(dim)//2
            if real.size(dim) % 2 != 0:
                n_shift += 1  # for odd-sized images
            real = self.roll_n(real, axis=dim, n=n_shift)
            imag = self.roll_n(imag, axis=dim, n=n_shift)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

    def batch_ifftshift2d(self, x):
        real, imag = torch.unbind(x, -1)
        for dim in range(len(real.size()) - 1, 0, -1):
            real = self.roll_n(real, axis=dim, n=real.size(dim)//2)
            imag = self.roll_n(imag, axis=dim, n=imag.size(dim)//2)
        return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

