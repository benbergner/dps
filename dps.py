import torch
from torch import nn

from resnet import resnet18
from transformer import Transformer


class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 500, sigma: float = 0.05):
        super(PerturbedTopK, self).__init__()
    
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x):
        return PerturbedTopKFunction.apply(x, self.k, self.num_samples, self.sigma)


class PerturbedTopKFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 500, sigma: float = 0.05):
        b, d = x.shape
        # for Gaussian: noise and gradient are the same.
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(x.device)

        perturbed_x = x[:, None, :] + noise * sigma # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices # b, nS, k
        indices = torch.sort(indices, dim=-1).values # b, nS, k

        # b, nS, k, d
        perturbed_output = torch.nn.functional.one_hot(indices, num_classes=d).float()
        indicators = perturbed_output.mean(dim=1) # b, k, d

        # constants for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        # tensors for backward
        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (
            torch.einsum("bnkd,bne->bkde", ctx.perturbed_output, noise_gradient)
            / ctx.num_samples
            / ctx.sigma
        ) * float(ctx.k)
        grad_input = torch.einsum("bkd,bkde->be", grad_output, expected_gradient)
        
        return (grad_input,) + tuple([None] * 5)


class Scorer(nn.Module):
    ''' Scorer network '''

    def __init__(self, n_channel):
        # Initialize the base class
        super().__init__()    
        
        # Create a ResNet-18 model with 3 channels and pretrained weights
        resnet = resnet18(num_channels=3, pretrained=True)
        
        # Define the scorer as a sequence of layers from the ResNet model,
        # followed by a Conv2d layer with 1 output channel and a kernel size of 3,
        # and a MaxPool2d layer with a kernel size of (2,2) and a stride of (2,2)
        self.scorer = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            nn.Conv2d(128, 1, kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
        
        # Delete the ResNet model as it is not needed anymore
        del resnet
    
    def forward(self):
        # Pass the input through the scorer layers
        x = self.scorer(x)
        
        # Squeeze the channel dimension
        x = x.squeeze(1)
        
        # Return the scored input
        return x


class DPS(nn.Module):
    ''' Differentiable Patch Selection '''

    def __init__(self, task, n_class, n_channel, k, patch_size, n_layer, n_token, n_head,
        d_k, d_v, d_model, d_inner, dropout, attn_dropout, device, use_pretrained=True, use_pos_enc=False):
        super().__init__()

        self.patch_size = patch_size
        self.k = k
        self.device = device

        self.scorer = Scorer(n_channel)

        self.TOPK = PerturbedTopK(k=k)
        self.feature_net = resnet18(num_channels=n_channel, pretrained=use_pretrained, flatten=True)

        self.transformer = Transformer(n_layer, n_token, n_head, d_k, d_v, d_model, d_inner, attn_dropout, dropout)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, n_class),
            nn.Softmax(dim = -1)
        )
    
    def forward(self, x_high, x_low):
        
        b, c, h_orig, w_orig = x_high.shape
        patch_size = self.patch_size
        device = self.device

        ### Score patches to get indicators
        scores_2d = self.scorer(x_low)
        b, h_score, w_score = scores_2d.shape
        scores_1d = scores_2d.view(b, -1)

        # entropy
        prob_scores = torch.softmax(scores_1d, dim=-1)
        entr = torch.special.entr(prob_scores).sum(-1).mean(0)

        # 0 -1 normalization
        scores_min = scores_1d.min(axis=-1, keepdims=True)[0]
        scores_max = scores_1d.max(axis=-1, keepdims=True)[0]
        scores_1d =  (scores_1d - scores_min) / (scores_max - scores_min + 1e-5)

        indicators = self.TOPK(scores_1d).view(b, self.k, h_score, w_score)
    
        ### Extract patches
        # Pad image      
        scale_h = h_orig // h_score
        scale_w = w_orig // w_score
        padded_h = scale_h * h_score + patch_size - 1
        padded_w = scale_w * w_score + patch_size - 1
        top_pad = (patch_size - scale_h) // 2
        left_pad = (patch_size - scale_w) // 2
        bottom_pad = padded_h - top_pad - h_orig
        right_pad = padded_w - left_pad - w_orig

        padding = (left_pad, right_pad, top_pad, bottom_pad)
        x_high_pad = torch.nn.functional.pad(x_high, padding, "constant", 0)

        patches = torch.zeros((b, k, c, patch_size, patch_size)).to(device)
        for i in range(h_score):
            for j in range(w_score):
                start_h = i*scale_h
                start_w = j*scale_w

                current_patches = x_high_pad[:, :, start_h : start_h + patch_size , start_w : start_w + patch_size] #(b, c, patch_size, patch_size)
                weight = indicators[:, :, i, j] #b, k

                patches += torch.einsum('bchw,bk->bkchw', current_patches, weight) #broacast, element-wise mult.
        
        ### Extract features, aggregate, predict
        features = self.feature_net(patches.view(-1, c, self.patch_size, self.patch_size)).view(b, self.k, -1)
        features = self.transformer(features)
        pred = self.head(features).view(b, -1)

        return pred, entr