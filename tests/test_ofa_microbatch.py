import torch
from research.image_attribution.models import ofa_features


def test_ofa_denoiser_uses_fixed_microbatch_for_single_batch_parity():
    class Denoiser(torch.nn.Module):
        def __init__(self):super().__init__();self.sizes=[]
        def forward(self,x):self.sizes.append(len(x));return x+len(x)*.01
    denoiser=Denoiser();encoder=torch.nn.Identity()
    x=torch.zeros(3,3,4,4)
    batch=ofa_features(x,encoder,denoiser)
    single=ofa_features(x[:1],encoder,denoiser)
    assert torch.equal(batch[0],single[0])
    assert denoiser.sizes==[1,1,1,1]
