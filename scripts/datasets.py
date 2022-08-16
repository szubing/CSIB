import numpy as np
import torch
import math


class ExampleMy1:
    """
    Cows and camels
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "binary_classification"
        self.envs = {}

        if n_envs >= 1:
            self.envs['E0'] = {"p": 0.95, "s": 0.5}

        if n_envs >= 2:
            self.envs['E1'] = {"p": 0.97, "s": 0.7}
        
        if n_envs >= 3:
            self.envs["E2"] = {"p": 0.99, "s": 0.3}

        if n_envs > 3:
            for env in range(3, n_envs):
                self.envs["E" + str(env)] = {
                    "p": torch.zeros(1).uniform_(0.9, 1).item(),
                    "s": torch.zeros(1).uniform_(0.3, 0.7).item()
                }
        print("Environments variables:", self.envs)

        # foreground is 100x noisier than background
        self.snr_fg = 1e-2
        self.snr_bg = 1

        # foreground (fg) denotes animal (cow / camel)
        cow = torch.ones(1, self.dim_inv)
        self.avg_fg = torch.cat((cow, cow, -cow, -cow))

        # background (bg) denotes context (grass / sand)
        grass = torch.ones(1, self.dim_spu)
        self.avg_bg = torch.cat((grass, -grass, -grass, grass))

        # adding noise to invariant features
        self.noise = 0  # try 0.05 0.1 for more experiments
        self.make_spu_separable = True # try False for interesting results to use_backbone & not use_backone

    def sample(self, n=1000, env="E0", split="train"):
        if self.make_spu_separable:
            p = 1.0 # for My test use
        else:
            p = self.envs[env]["p"]
        # 
        s = self.envs[env]["s"]
        w = torch.Tensor([p, 1 - p] * 2) * torch.Tensor([s] * 2 + [1 - s] * 2)
        i = torch.multinomial(w, n, True)
        x = torch.cat((
            (torch.randn(n, self.dim_inv) /
                math.sqrt(10) + self.avg_fg[i]) * self.snr_fg,
            (torch.randn(n, self.dim_spu) /
                math.sqrt(10) + self.avg_bg[i]) * self.snr_bg), -1)

        if split == "test":
            x[:, self.dim_spu:] = x[torch.randperm(len(x)), self.dim_spu:]

        inputs = x @ self.scramble
        outputs = x[:, :self.dim_inv].sum(1, keepdim=True).gt(0)

        if self.noise > 0:
            flag = torch.rand(n,1) < self.noise
            outputs = outputs ^ flag
        
        outputs = outputs.float()

        return inputs, outputs


class ExampleMy2:
    """
    Z_inv = A Z_spu + W
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu
        assert dim_inv == dim_spu

        self.task = "binary_classification"
        self.envs = {}

        if n_envs >= 1:
            self.envs['E0'] = {"a": 1.0, "z_spu": 5.0, "w": 3.0}
        if n_envs >= 2:
            self.envs['E1'] = {"a": 1.0, "z_spu": 5.0, "w": 2.0}
        if n_envs >= 3:
            self.envs["E2"] = {"a": 1.0, "z_spu": 5.0, "w": 1.0}
        if n_envs > 3:
            for env in range(3, n_envs):
                self.envs["E" + str(env)] = {
                    "a": 1.0,
                    "z_spu": 5.0,
                    "w": torch.zeros(1).uniform_(0, 3).item()
                }
        print("Environments variables:", self.envs)

        self.spu_ = 1e-2
        self.w_ = 1

    def sample(self, n=1000, env="E0", split="train"):
        a = self.envs[env]["a"]
        z_spu = self.envs[env]["z_spu"]
        w = self.envs[env]["w"]
        
        if split == "test":
            # x[:, self.dim_spu:] = x[torch.randperm(len(x)), self.dim_spu:]
            a = 1.0
            z_spu = 5.0
            w = 10.0
        
        # A = torch.randn(self.dim_inv, self.dim_spu) * a
        A = torch.eye(self.dim_spu) * a
        Z_spu = torch.cat((torch.randn(n//2, self.dim_spu) * self.spu_ + z_spu, torch.randn(n//2, self.dim_spu) * self.spu_ - z_spu))
        Z_spu = Z_spu[torch.randperm(len(Z_spu)), :]
        W = torch.cat((torch.randn(n//2, self.dim_spu) * self.w_ + w, torch.randn(n//2, self.dim_spu) * self.w_ - w))
        W = W[torch.randperm(len(W)), :]
        Z_inv = Z_spu + W
        x = torch.cat((Z_inv, Z_spu), -1)

        inputs = x @ self.scramble
        outputs = x[:, :self.dim_inv].sum(1, keepdim=True).gt(0).float()

        return inputs, outputs


class ExampleMy3:
    """
    Small invariant margin versus large spurious margin
    """

    def __init__(self, dim_inv, dim_spu, n_envs):
        self.scramble = torch.eye(dim_inv + dim_spu)
        self.dim_inv = dim_inv
        self.dim_spu = dim_spu
        self.dim = dim_inv + dim_spu

        self.task = "binary_classification"
        self.envs = {}

        for env in range(n_envs):
            self.envs["E" + str(env)] = torch.zeros(1).uniform_(0, 1).item()
        
        print("Environments variables:", self.envs)

    def sample(self, n=1000, env="E0", split="train"):
        m = n // 2
        sep = 10.0
        inv_ = 10.0
        spu_ = self.envs[env]
        if split == "test":
            spu_ = 10

        invariant = torch.cat((torch.randn(m, self.dim_inv) * inv_ + sep, torch.randn(m, self.dim_inv) * inv_ - sep))
        invariant = invariant[torch.randperm(len(invariant)), :]
        outputs = invariant[:, :self.dim_inv].sum(1, keepdim=True).gt(0).float()

        spurious = torch.cat((2*outputs[:m] @ torch.ones(1, self.dim_spu) - 1 + spu_, 2*outputs[m:] @ torch.ones(1, self.dim_spu) - 1 - spu_ ))

        x = torch.cat((invariant, spurious), -1)

        inputs = x @ self.scramble

        return inputs, outputs


class ExampleMy1s(ExampleMy1):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))

class ExampleMy2s(ExampleMy2):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))

class ExampleMy3s(ExampleMy3):
    def __init__(self, dim_inv, dim_spu, n_envs):
        super().__init__(dim_inv, dim_spu, n_envs)

        self.scramble, _ = torch.qr(torch.randn(self.dim, self.dim))


DATASETS = {
    "ExampleMy1": ExampleMy1,
    "ExampleMy2": ExampleMy2,
    "ExampleMy3": ExampleMy3,
    "ExampleMy1s": ExampleMy1s,
    "ExampleMy2s": ExampleMy2s,
    "ExampleMy3s": ExampleMy3s,
}