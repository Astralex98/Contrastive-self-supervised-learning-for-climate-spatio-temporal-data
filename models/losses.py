import torch
from torch import nn
import torch.nn.functional as F

class Our_TS2Vec_loss(torch.nn.Module):
    
    def __init__(self, alpha=0.5, temporal_unit=0, tau=0.1):
        super(Our_TS2Vec_loss, self).__init__()
        
        self.alpha = alpha
        self.temporal_unit = temporal_unit
        self.tau = tau

    def instance_contrastive_loss(self, z1, z2):
        
        B, T = z1.size(0), z1.size(1)
        
        if B == 1:
            return z1.new_tensor(0.)
        
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:] # Removing similarities between same samples in the batch, like (B[1], B[1])
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z.device)
        
        # Here we choose only similarities between augmentations like (B[1], B'[1]), (B[2], B'[2]), ... (first term)
        # and averaging first term across batch and time dimension
        # Likewise for the second term where we choose another pairs like (B'[1], B[1]), (B'[2], B[2])
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    # Stable version of softmax function
    # logits: similarity matrix B x T x T
    # To make log softmax numerical stable I used formula from here:
    # https://stackoverflow.com/questions/61567597/how-is-log-softmax-implemented-to-compute-its-value-and-gradient-with-better
    def log_softmax_temporal(self, logits, epsilon = 1e-5):
        
        for ind_region in range(logits.shape[0]):
            
            const = torch.max(logits[ind_region, :, :]/self.tau)
            
            for ind_x in range(logits.shape[1]):
                
                denumerator = torch.sum(torch.exp(logits[ind_region, ind_x, (ind_x + 1):logits.shape[2]]/self.tau - const))
                
                for ind_y in range(ind_x + 1, logits.shape[2]):
                        
                    # add small epsilon to logarithm to avoid explosion (ln(0)=-inf)
                    loss = const + torch.log(denumerator + epsilon) - (logits[ind_region, ind_x, ind_y] / self.tau)
                    
                    if (torch.isinf(loss) == True):
                        print("const: {}".format(const))
                        print("log denum: {}".format(denumerator))
                        print("val: {}".format(logits[ind_region, ind_x, ind_y] / self.tau))
                        print("loss: {}".format(loss))
                        
                    logits[ind_region, ind_x, ind_y] = loss
                    logits[ind_region, ind_y, ind_x] = loss
        
        return logits
    
    def temporal_contrastive_loss(self, z1, z2):
        
        # z - embeddings of original data
        B, T = z1.size(0), z1.size(1)
        if T == 1:
            return z1.new_tensor(0.)
        
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        sim = torch.matmul(z, z.transpose(1, 2))  # 2B x T x T
        sim = self.log_softmax_temporal(sim) # 2B x T x T
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # 2B x T x (T-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        
        loss = logits.mean()
        return loss

    def forward(self, z_orig, z_augs):
        
        loss = torch.tensor(0., device=z_orig.device)
        d = 0
        while z_orig.size(1) > 1:
            if self.alpha != 0:
                loss += self.alpha * self.instance_contrastive_loss(z_orig, z_augs)
            if d >= self.temporal_unit:
                if 1 - self.alpha != 0:
                    loss += (1 - self.alpha) * self.temporal_contrastive_loss(z_orig, z_augs)
            d += 1
            z_orig = F.max_pool1d(z_orig.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z_augs = F.max_pool1d(z_augs.transpose(1, 2), kernel_size=2).transpose(1, 2)
        
        if z_orig.size(1) == 1:
            if self.alpha != 0:
                loss += self.alpha * self.instance_contrastive_loss(z_orig, z_augs)
            d += 1

        return loss / d
    
    
    
    
    
    
    
######################################################TS2Vec#######################################################################

class TS2Vec_loss(torch.nn.Module):
    
    def __init__(self, alpha=0.5, temporal_unit=0):
        super(TS2Vec_loss, self).__init__()
        self.alpha = alpha
        self.temporal_unit = temporal_unit

    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        if B == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:] # Removing similarities between same samples in the batch, like (B[1], B[1])
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        
        # Here we choose only similarities between augmentations like (B[1], B'[1]), (B[2], B'[2]), ... (first term)
        # and averaging first term across batch and time dimension
        # Likewise for the second term where we choose another pairs like (B'[1], B[1]), (B'[2], B[2])
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def temporal_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        if T == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=1)  # B x 2T x C
        sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        return loss

    def forward(self, z1, z2):
        loss = torch.tensor(0., device=z1.device)
        d = 0
        while z1.size(1) > 1:
            if self.alpha != 0:
                loss += self.alpha * self.instance_contrastive_loss(z1, z2)
            if d >= self.temporal_unit:
                if 1 - self.alpha != 0:
                    loss += (1 - self.alpha) * self.temporal_contrastive_loss(z1, z2)
            d += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        if z1.size(1) == 1:
            if self.alpha != 0:
                loss += self.alpha * self.instance_contrastive_loss(z1, z2)
            d += 1

        return loss / d