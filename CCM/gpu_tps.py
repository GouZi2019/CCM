import math
import torch
import torch.nn.functional as F

import SimpleITK as sitk
import numpy as np

class TPS:       
    @staticmethod
    def fit(c, f, lambd=0.):
        device = c.device
        
        n = c.shape[0]
        d = f.shape[1]

        U = TPS.u(TPS.d(c, c))
        K = U + torch.eye(n, device=device) * lambd

        P = torch.ones((n, (d+1)), device=device)
        P[:, 1:] = c

        v = torch.zeros((n+(d+1), d), device=device)
        v[:n, :] = f

        A = torch.zeros((n+(d+1), n+(d+1)), device=device)
        A[:n, :n] = K
        A[:n, -(d+1):] = P
        A[-(d+1):, :n] = P.t()

        theta = torch.linalg.solve(A, v)
        return theta
        
    @staticmethod
    def d(a, b):
        ra = (a**2).sum(dim=1).view(-1, 1)
        rb = (b**2).sum(dim=1).view(1, -1)
        dist = ra + rb - 2.0 * torch.mm(a, b.permute(1, 0))
        dist.clamp_(0.0, float('inf'))
        return torch.sqrt(dist)

    @staticmethod
    def u(r):
        return r

    @staticmethod
    def z(x, c, theta):
        d = c.shape[1]

        U = TPS.u(TPS.d(x, c))
        w, a = theta[:-(d+1)], theta[-(d+1):].unsqueeze(-1)
        b = torch.matmul(U, w)

        if d == 2:
            return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + b.t()).t()
        else:
            return (a[0] + a[1] * x[:, 0] + a[2] * x[:, 1] + a[3] * x[:, 2] + b.t()).t()

def gpu_tps(pfs, pms, reference: sitk.Image=None, lambd=.0, unroll_step_size=2**12):
    d = reference.GetDimension()
    N = reference.GetNumberOfPixels()

    pfs = torch.Tensor(pfs).cuda()
    pms = torch.Tensor(pms).cuda()

    img_origin = np.array(reference.GetOrigin()).reshape(d,1)
    img_spacing = np.array(reference.GetSpacing())
    img_direction = np.array(reference.GetDirection()).reshape(d,d)

    img_index2world = np.dot(img_direction, np.diag(img_spacing))
    grid_index = np.array(np.unravel_index(range(N), reference.GetSize()[::-1]))[::-1]
    grid_world = np.dot(img_index2world, grid_index) + img_origin
    grid_world = torch.Tensor(grid_world).cuda().T

    tps = TPS()
    theta = tps.fit(pfs, pms-pfs, lambd)

    y = torch.zeros((N, d)).cuda()
    n = math.ceil(N/unroll_step_size)
    for j in range(n):
        j1 = j * unroll_step_size
        j2 = min((j + 1) * unroll_step_size, N)
        y[j1:j2, :] = tps.z(grid_world[j1:j2], pfs, theta)
    
    output_size = list(reference.GetSize()[::-1]) + [d]
    field_array = y.view(output_size).cpu().numpy()
    field = sitk.GetImageFromArray(field_array, isVector=True)
    field.CopyInformation(reference)
    
    return sitk.Cast(field, sitk.sitkVectorFloat64)