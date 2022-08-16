import torch
import json
import random
import numpy as np
import utils
from torch.autograd import grad
import pdb


CPU = torch.device("cpu")


class Model(torch.nn.Module):
    def __init__(self, in_features, out_features, task="binary_classification", hparams="default", device=CPU, use_backbone=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.task = task
        self.use_backbone = use_backbone
        self.device = device

        # hyper-parameters
        if hparams == "default":
            self.hparams = {k: v[0] for k, v in self.HPARAMS.items()}
        elif hparams == "random":
            self.hparams = {k: v[1] for k, v in self.HPARAMS.items()}
        else:
            self.hparams = json.loads(hparams)

        if not self.use_backbone:
            # network architecture
            self.network = torch.nn.Linear(in_features, out_features)
            self.network = self.network.to(device)
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["wd"])
        else:
            # network architecture with backbone
            self.backbone = torch.nn.Linear(in_features, in_features, bias=False)
            self.classifier = torch.nn.Linear(in_features, out_features)
            self.backbone = self.backbone.to(device)
            self.classifier = self.classifier.to(device)
            self.optimizer = torch.optim.Adam(
                [{'params': self.backbone.parameters()}, 
                 {'params': self.classifier.parameters()}],
                lr=self.hparams["lr"],
                weight_decay=self.hparams["wd"])
        

        # loss
        if self.task == "binary_classification":
            self.loss = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss = torch.nn.CrossEntropyLoss()

        # callbacks
        self.callbacks = {}
        for key in ["errors"]:
            self.callbacks[key] = {
                "train": [],
                "validation": [],
                "test": []
            }


class ERM(Model):
    def __init__(self, in_features, out_features, task, hparams="default", device=CPU, use_backbone=True):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))

        super().__init__(in_features, out_features, task, hparams, device, use_backbone)

    def fit(self, envs, num_iterations, callback=False):
        x = torch.cat([xe for xe, ye in envs["train"]["envs"]])
        y = torch.cat([ye for xe, ye in envs["train"]["envs"]])

        x, y = x.to(self.device), y.to(self.device)
        for epoch in range(num_iterations):
            self.optimizer.zero_grad()
            if self.use_backbone:
                out = self.classifier(self.backbone(x))
            else:
                out = self.network(x)

            self.loss(out, y).backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        if self.use_backbone:
            return self.classifier(self.backbone(x))
        else:
            return self.network(x)


class IB_ERM(Model):
    def __init__(self, in_features, out_features, task, hparams="default", device=CPU, use_backbone=True):
        # self.args = args
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))

        self.HPARAMS['ib_lambda'] = (1.0, 1 - 10**random.uniform(-2, 0.0))  # v1 

        super().__init__(in_features, out_features, task, hparams, device, use_backbone)

    def fit(self, envs, num_iterations, callback=False):
        x = torch.cat([xe for xe, ye in envs["train"]["envs"]])
        y = torch.cat([ye for xe, ye in envs["train"]["envs"]])

        x, y = x.to(self.device), y.to(self.device)
        for epoch in range(num_iterations):
            self.optimizer.zero_grad()
            if self.use_backbone:
                feature_logits = self.backbone(x)
                logits = self.classifier(feature_logits)
                loss = self.loss(logits, y)
                loss += self.hparams['ib_lambda'] * feature_logits.var(0).mean()
            else:
                logits = self.network(x)  # (30000, 1)
                loss = self.loss(logits, y)
                loss += self.hparams["ib_lambda"] * logits.var(0).mean()

            loss.backward()
            self.optimizer.step()

            if callback:
                utils.compute_errors(self, envs)

    def predict(self, x):
        if self.use_backbone:
            return self.classifier(self.backbone(x))
        else:
            return self.network(x)


class IIB(Model):
    def __init__(self, in_features, out_features, task, hparams="default", device=CPU):
        # self.args = args
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['ib_lambda'] = (1.0, 1 - 10**random.uniform(-2, 0.0))  # v1  # change default 0.1 to 1.0 get good result

        super().__init__(in_features, out_features, task, hparams, device, use_backbone=True)

        self.F = torch.eye(self.in_features).to(self.device)
        self.V = []
        self.r = []
    
    def set_scramble(self, scramble):
        self.scramble = scramble.to(self.device)

    def set_dim_inv(self, dim_inv):
        self.dim_inv = dim_inv

    def fit(self, envs, num_iterations, callback=False):
        assert self.use_backbone, 'IIB method requires a backbone feature extractor'
        x, y = self._fit(envs, num_iterations)
        Phi = self.backbone.state_dict()['weight'].detach()
        U, S, V = torch.svd(Phi)
        S_prob = S/torch.sum(S)
        r = len(S_prob)
        for i in range(1, len(S_prob)-1):
            if (S_prob[i] / S_prob[i+1]) > (100 * S_prob[i-1] / S_prob[i]):
                r = i + 1
                break
            elif (100* S_prob[i] / S_prob[i+1]) < (S_prob[i-1] / S_prob[i]):
                r = i
                break
        
        if r < len(V):
            m = 100.0
            f = x @ self.F.T @ V
            f1 = f.clone()
            f2 = f.clone()
            f1[0:r] = torch.ones(r) * m
            f2[0:r] = torch.ones(r) * -m
            f_new_1 = f1 @ V.T
            f_new_2 = f2 @ V.T

            f_init = x.clone()
            f_old = []
            for i in range(len(self.V)):
                current_f = f_init @ self.V[i]
                f_old.append(current_f.clone())
                f_init = current_f[self.r[i]:]

            for i in range(len(self.V)):
                j = len(self.V) - i - 1
                f_1 = f_old[j].clone()
                f_2 = f_old[j].clone()
                f_1[self.r[j]:] = f_new_1.clone()
                f_2[self.r[j]:] = f_new_2.clone()
                f_new_1 = f_1 @ self.V[j].T
                f_new_2 = f_2 @ self.V[j].T
            
            x_new_1 = f_new_1 @ self.scramble.T
            x_new_2 = f_new_2 @ self.scramble.T
            print('intervention data1: ', x_new_1, '-- intervention data2: ', x_new_2)
            y_new_1 = x_new_1[:self.dim_inv].sum().gt(0).float()
            y_new_2 = x_new_2[:self.dim_inv].sum().gt(0).float()
            if y_new_1 == y_new_2:
                self.backbone = torch.nn.Linear(len(V)-r, len(V)-r, bias=False)
                self.classifier = torch.nn.Linear(len(V)-r, self.out_features)
                self.backbone = self.backbone.to(self.device)
                self.classifier = self.classifier.to(self.device)
                self.optimizer = torch.optim.Adam(
                    [{'params': self.backbone.parameters()}, 
                        {'params': self.classifier.parameters()}],
                    lr=self.hparams["lr"],
                    weight_decay=self.hparams["wd"])
                self.F = V.T[r:] @ self.F
                self.r.append(r)
                self.V.append(V.clone())
                self.fit(envs, num_iterations)

        if callback:
            utils.compute_errors(self, envs)


    def _fit(self, envs, num_iterations):
        x = torch.cat([xe for xe, ye in envs["train"]["envs"]])
        y = torch.cat([ye for xe, ye in envs["train"]["envs"]])

        x, y = x.to(self.device), y.to(self.device)
        x_init, y_init = x[0], y[0]
        x = x @ self.F.T
        for epoch in range(num_iterations):
            self.optimizer.zero_grad()
            feature_logits = self.backbone(x)
            logits = self.classifier(feature_logits)
            loss = self.loss(logits, y)

            # entropy minimization loss
            loss += self.hparams['ib_lambda'] * feature_logits.var(0).mean()

            loss.backward()
            self.optimizer.step()

        return x_init, y_init

    def predict(self, x):
        return self.classifier(self.backbone(x@self.F.T))


class IRM(Model):
    """
    Abstract class for IRM
    """

    def __init__(
            self, in_features, out_features, task, hparams="default", device=CPU, version=1, use_backbone=True):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(-3, -.3))

        super().__init__(in_features, out_features, task, hparams, device, use_backbone)
        self.version = version

        if self.use_backbone:
            self.network = torch.nn.Sequential(self.backbone, self.classifier)

        self.network = self.IRMLayer(self.network).to(self.device)
        self.net_parameters, self.net_dummies = self.find_parameters(
            self.network)

    def find_parameters(self, network):
        """
        Alternative to network.parameters() to separate real parameters
        from dummmies.
        """
        parameters = []
        dummies = []

        for name, param in network.named_parameters():
            if "dummy" in name:
                dummies.append(param)
            else:
                parameters.append(param)
        return parameters, dummies

    class IRMLayer(torch.nn.Module):
        """
        Add a "multiply by one and sum zero" dummy operation to
        any layer. Then you can take gradients with respect these
        dummies. Often applied to Linear and Conv2d layers.
        """

        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.dummy_mul = torch.nn.Parameter(torch.Tensor([1.0]))
            self.dummy_sum = torch.nn.Parameter(torch.Tensor([0.0]))

        def forward(self, x):
            return self.layer(x) * self.dummy_mul + self.dummy_sum

    def fit(self, envs, num_iterations, callback=False):
        for epoch in range(num_iterations):
            losses_env = []
            gradients_env = []
            for x, y in envs["train"]["envs"]:
                x, y = x.to(self.device), y.to(self.device)
                losses_env.append(self.loss(self.network(x), y))
                gradients_env.append(grad(
                    losses_env[-1], self.net_dummies, create_graph=True))

            # Average loss across envs
            losses_avg = sum(losses_env) / len(losses_env)

            penalty = 0
            for gradients_this_env in gradients_env:
                for g_env in gradients_this_env:
                    penalty += g_env.pow(2).sum()

            obj = (1 - self.hparams["irm_lambda"]) * losses_avg
            obj += self.hparams["irm_lambda"] * penalty

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            if callback:
                # compute errors
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


class IRMv1(IRM):
    """
    IRMv1 with penalty \sum_e \| \nabla_{w|w=1} \mR_e (\Phi \circ \vec{w}) \|_2^2
    From https://arxiv.org/abs/1907.02893v1 
    """

    def __init__(self, in_features, out_features, task, hparams="default", device=CPU, use_backbone=True):
        super().__init__(in_features, out_features, task, hparams, device, 1, use_backbone)


class IB_IRM(Model):
    def __init__(
            self, in_features, out_features, task, hparams="default", device=CPU, version=1, use_backbone=True):
        self.HPARAMS = {}
        self.HPARAMS["lr"] = (1e-3, 10**random.uniform(-4, -2))
        self.HPARAMS['wd'] = (0., 10**random.uniform(-6, -2))
        self.HPARAMS['irm_lambda'] = (0.9, 1 - 10**random.uniform(-3, -.3))  # 0.5 ~ 0.999

        self.HPARAMS['ib_lambda'] = (1.0, 1 - 10**random.uniform(-2, 0))
        # self.HPARAMS['ib_on'] = (True, random.choice([True, False]))

        super().__init__(in_features, out_features, task, hparams, device, use_backbone)
        self.version = version

        if self.use_backbone:
            self.network = torch.nn.Sequential(self.backbone, self.classifier)

        self.network = self.IRMLayer(self.network).to(self.device)
        self.net_parameters, self.net_dummies = self.find_parameters(
            self.network)

        self.optimizer = torch.optim.Adam(
            self.net_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["wd"])

    def find_parameters(self, network):
        """
        Alternative to network.parameters() to separate real parameters
        from dummmies.
        """
        parameters = []
        dummies = []

        for name, param in network.named_parameters():
            if "dummy" in name:
                dummies.append(param)
            else:
                parameters.append(param)
        return parameters, dummies

    class IRMLayer(torch.nn.Module):

        def __init__(self, layer):
            super().__init__()
            self.layer = layer
            self.dummy_mul = torch.nn.Parameter(torch.Tensor([1.0]))
            self.dummy_sum = torch.nn.Parameter(torch.Tensor([0.0]))

        def forward(self, x):
            return self.layer(x) * self.dummy_mul + self.dummy_sum

    def fit(self, envs, num_iterations, callback=False):
        for epoch in range(num_iterations):
            losses_env = []
            gradients_env = []
            logits_env = []
            for x, y in envs["train"]["envs"]:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.network(x)
                logits_env.append(logits)
                losses_env.append(self.loss(self.network(x), y))
                gradients_env.append(grad(losses_env[-1], self.net_dummies, create_graph=True))

            # penalty per env
            # torch.stack(logits_env): (3, 10000, 1)
            logit_penalty = torch.stack(logits_env).var(1).mean()

            # Average loss across envs
            losses_avg = sum(losses_env) / len(losses_env)

            penalty = 0
            for gradients_this_env in gradients_env:
                for g_env in gradients_this_env:
                    penalty += g_env.pow(2).sum()

            obj = (1 - self.hparams["irm_lambda"]) * losses_avg
            obj += self.hparams["irm_lambda"] * penalty

            # if self.hparams['ib_on'] or (not self.args["ib_bool"]):
            obj += self.hparams["ib_lambda"] * logit_penalty

            self.optimizer.zero_grad()
            obj.backward()
            self.optimizer.step()

            if callback:
                utils.compute_errors(self, envs)

    def predict(self, x):
        return self.network(x)


MODELS = {
    "ERM": ERM,
    # "IRMv1": IRMv1,
    "IRM": IRM,
    "Oracle": ERM,
    "IB_ERM": IB_ERM,
    "IB_IRM": IB_IRM,
    "IIB": IIB
}

