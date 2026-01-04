import torch
import torch.nn.functional as F
from torch.fft import ifft2, fft2, fftshift

class Fresnel_diffraction():

    def __init__(self, Z, pix, Energy, ratio, cuda_device):

        self.Z = Z  # propagation distance: mm
        self.pixel_size = pix  # detector pixel size: mm
        self.E = Energy    # energy: keV
        self.ratio = ratio  # delta/beta
        self.device = cuda_device

    def forward(self, phase):

        absorp = phase / self.ratio

        lambda_ = 1e-6 * (1.24 / self.E)  # x-ray wavelength: mm
        k = 2 * torch.pi / lambda_  # wave number

        T0 = torch.exp(- absorp / 2 - 1j * phase)

        rows = T0.shape[-2]
        cols = T0.shape[-1]

        F_fft = fft2(F.pad(T0, (cols // 2, cols // 2, rows // 2, rows // 2), mode='reflect'))

        fs = 1 / self.pixel_size
        Kx, Ky = torch.meshgrid(torch.arange(-fs / 2, fs / 2, fs / (rows * 2)),
                                torch.arange(-fs / 2, fs / 2, fs / (cols * 2)))

        fresnel_fft = fftshift(torch.exp(torch.as_tensor(1j * k * self.Z)) * torch.exp(
            torch.as_tensor(-1j * torch.pi * lambda_ * self.Z * (Kx ** 2 + Ky ** 2)))).to(self.device)

        U_Delta = ifft2(F_fft * fresnel_fft)

        U_Delta = U_Delta[:, :, rows // 2: 3 * rows // 2, cols // 2:3 * cols // 2]

        return (torch.abs(U_Delta) ** 2).to(dtype=torch.float32)