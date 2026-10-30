"""Per-image full-reference RGB metrics, pixels in [0,1], no border crop."""
import torch
import torch.nn.functional as F


def image_metrics(prediction, target):
    prediction=prediction.float();target=target.float()
    if prediction.shape!=target.shape or prediction.ndim!=4 or prediction.shape[0]!=1:
        raise ValueError('Metrics require one aligned RGB image')
    mse=(prediction-target).square().mean()
    size=min(11,*prediction.shape[-2:]);size-=1-size%2
    axis=torch.arange(size,device=prediction.device,dtype=torch.float32)-(size-1)/2
    gaussian=torch.exp(-axis.square()/(2*1.5**2));gaussian/=gaussian.sum()
    kernel=(gaussian[:,None]*gaussian[None,:]).expand(3,1,size,size).contiguous()
    def smooth(value):return F.conv2d(value,kernel,groups=3)
    mu_x=smooth(prediction);mu_y=smooth(target)
    var_x=smooth(prediction.square())-mu_x.square()
    var_y=smooth(target.square())-mu_y.square()
    covariance=smooth(prediction*target)-mu_x*mu_y
    ssim=(((2*mu_x*mu_y+.01**2)*(2*covariance+.03**2))/
          ((mu_x.square()+mu_y.square()+.01**2)*(var_x+var_y+.03**2))).mean()
    gradients=[]
    for dimension in (-1,-2):
        if prediction.shape[dimension]>1:
            gradients.append((prediction.diff(dim=dimension)-target.diff(dim=dimension)).abs().mean())
    return {'l1':float((prediction-target).abs().mean()),'mse':float(mse),
            'psnr':float(-10*mse.clamp_min(1e-12).log10()),'ssim':float(ssim),
            'gradient_l1':float(torch.stack(gradients).mean()) if gradients else 0.}
