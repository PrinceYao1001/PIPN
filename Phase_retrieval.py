import torch
from Data import ProjectionsImage
from Fresnel_Diffraction import Fresnel_diffraction
from Backbone import U_Net
from utils import TensorNorm
from torch.utils.data import  DataLoader
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
import configargparse

p = configargparse.ArgumentParser()
# Data path
p.add_argument('--projection_path', default='/media/omnisky/data/wzy/WorkData/PhaseRetrieval',
               type=str, help='Path to the raw projections.')
p.add_argument('--saving_path', default='/media/omnisky/data/wzy/WorkData/PhaseRetrieval/results',
               type=str, help='Path to the results of the phase retrieval.')
# General phase retrieval options
p.add_argument('--batch_size', default=1, type=int,
               help='The results in the paper are based on a batch_size of 1.')
p.add_argument('--learning_rate', type=float, default='1e-3',)
p.add_argument('--cuda_index', type=int, default='0',
               help='Your cuda device index.')
p.add_argument('--initial_epochs', type=float, default=500)
p.add_argument('--saving_epoch', type=float, default=500,
               help='Save results after how many epochs')
p.add_argument('--Acceleration', type=bool, default=True,
               help='Whether or not to use acceleration strategy')
p.add_argument('--epochs_ac', type=float, default=200,
               help='Number of epochs required after acceleration')
p.add_argument('--Z', type=float, default=155,
               help='Propagation distance: mm')
p.add_argument('--pixel_size', type=float, default=3.25e-3,
               help='Detector pixel size: mm')
p.add_argument('--Energy', type=float, default=32,
               help='X-ray energy: Kev')
p.add_argument('--ratio', type=float, default=200,
               help='Delta / Beta')
opt = p.parse_args()

device = torch.device(f"cuda:{opt.cuda_index}")

raw_projection = ProjectionsImage(opt.projection_path)
projection_dataloader = tqdm(DataLoader(raw_projection, batch_size=opt.batch_size, shuffle=False, num_workers=1))
#   The input order of the raw projections must match the capture sequence
#   and must not be shuffled during input to enable the acceleration strategy.
fresnel_diffraction = Fresnel_diffraction(Z=opt.Z, pix=opt.pixel_size, Energy=opt.Energy, ratio=opt.ratio, cuda_device=device)


model = U_Net(in_ch=1, out_ch=1).to(device)

loss_func = torch.nn.L1Loss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

epochs = opt.initial_epochs

for index, proj in enumerate(projection_dataloader):

    projection_dataloader.set_description(f'Phase retrieving of View {index + 1}/{len(projection_dataloader)}')

    proj = proj.to(device)

    model.train()

    for epoch in range(epochs):

        optimizer.zero_grad()

        PR_projection = model(proj)
        PR_projection = TensorNorm(PR_projection)

        # Fresnel_propagation
        computed_projection = fresnel_diffraction.forward(PR_projection).to(device)

        computed_projection = TensorNorm(computed_projection)

        norm_projection = transforms.Normalize(mean=torch.mean(computed_projection),
                                              std=torch.std(computed_projection))(computed_projection)

        loss = loss_func(proj, norm_projection)
        loss.backward()

        projection_dataloader.set_postfix({'Loss':loss.item()})

        optimizer.step()

        if (epoch + 1) % opt.saving_epoch == 0:
            save_image(PR_projection, f'{opt.saving_path}/{index + 1}.tiff')

    if opt.Acceleration == True:
        epochs = opt.epochs_acc
