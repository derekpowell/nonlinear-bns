
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.special import logit, expit
import torch

def rescale_beta(x, lower=-3, upper=3):
    res = (x - lower) / (upper - lower)
    return((res * (len(x) - 1) + .5) / len(x))


def standardize(x):
    return( (x - np.mean(x))/np.std(x) )


def tiers2blacklist(tiers):
    total_nodes = len(tiers)
    adj = np.zeros((total_nodes, total_nodes))
    
    # i,j = 1 if tier i <= tier j
    for i in range(total_nodes):
        t_i = tiers[i]
        for j in range(total_nodes):
            t_j = tiers[j]
            if i != j:
                if t_i < t_j:
                    adj[i,j] = 1

    return(adj.T)



import matplotlib.pyplot as plt

def visualize_adj_matrix(mat, size=4.0):
    """    
    `mat`: (d, d) 
    """
    ## from DIBS library: https://github.com/larslorch/dibs/tree/master
    plt.rcParams['figure.figsize'] = [size, size]
    fig, ax = plt.subplots(1, 1)
    ax.matshow(mat, vmin=0, vmax=1)
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_title(r'Graph $G^*$', pad=10)
    plt.show()
    return


import torch
from dagma import utils
from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaMLP, DagmaNonlinear

class DagmaMLP2(DagmaMLP):

    def __init__(self, blackadj = None, **kwargs): # replace with 
        
        super().__init__(**kwargs)
        if blackadj is None:
            self.blackadj = torch.zeros(self.d, self.d).detach()
        else:
            self.blackadj = blackadj.detach()

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        r"""
        Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG

        Parameters
        ----------
        s : float, optional
            Controls the domain of M-matrices, by default 1.0

        Returns
        -------
        torch.Tensor
            A scalar value of the log-det acyclicity function :math:`h(\Theta)`.
        """
        
        fc1_weight = self.fc1.weight
        fc1_weight = fc1_weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight ** 2, dim=1).t()  # [i, j]
        h = -torch.slogdet(s * self.I - A)[1] + self.d * np.log(s) # +inf when not a DAG

        return h + ((A * self.blackadj).sum())*1e16


class DagmaLinear2(DagmaLinear):

    def predict(self, X = None):
        if X is None:
            X = self.X

        return X @ self.W_est


# class DagmaNonlinearBeta(DagmaNonlinear):

#     def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         r"""
#         Computes the logarithm of the MSE loss:
#             .. math::
#                 \frac{d}{2} \log\left( \frac{1}{n} \sum_{i=1}^n (\mathrm{output}_i - \mathrm{target}_i)^2 \right)
        
#         Parameters
#         ----------
#         output : torch.Tensor
#             :math:`(n,d)` output of the model
#         target : torch.Tensor
#             :math:`(n,d)` input dataset

#         Returns
#         -------
#         torch.Tensor
#             A scalar value of the loss.
#         """
#         logit = torch.sigmoid(output)
#         # l = torch.nn.KLDivLoss()
#         # loss = l(logit, target)

#         loss = torch.sum(Beta2(logit, 10.).log_prob(target))
#         return loss


def adj_to_edge_tuple(x):
    edge_list = np.stack(np.nonzero(x), axis=-1).tolist()
    return(tuple(tuple(sub) for sub in edge_list))


def Beta2(mu, k):
    return(torch.distributions.Beta(mu*k, (1.-mu)*k))