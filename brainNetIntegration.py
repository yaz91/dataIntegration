#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)


# In[2]:


import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid,TUDataset
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv,GAE,global_add_pool
from torch_geometric.utils import train_test_split_edges,from_networkx,negative_sampling
from torch_geometric.loader import DataLoader


# In[3]:


import numpy as np
import networkx


# In[4]:


import pickle
with open('../data/multiBrain.pickle','rb') as f:
    data = pickle.load(f)


# In[6]:


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


# In[7]:


class multiVGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.

    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, encoder, decoder=None, beta=4, lambdaMu = 0.001,modals = 3,MAX_LOGSTD=10):
        super(multiVGAE, self).__init__(encoder, decoder)
        self.betaReg = beta
        self.lambdaMu = lambdaMu
        self.modals = modals
        self.MAX_LOGSTD = MAX_LOGSTD
        self.EPS = 10**-5
        
        self.encoder = torch.nn.ModuleList(encoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, gData):
        """"""
        self.__mus__ = []
        self.__logstd__ = []
        for i in range(self.modals):
            mu, logstd = self.encoder[i](gData[i].x,gData[i].edge_index)
            # mu, logstd = self.encoder[i](gData[i].x,gData[i].edge_index,gData[i].batch)
            self.__mus__.append(mu)
            self.__logstd__.append(logstd.clamp(max=self.MAX_LOGSTD))
        self.__mu__,_ = torch.max(torch.stack(model.__mus__),0)
        z = []
        for i in range(self.modals):
            z.append(self.reparametrize(self.__mu__, self.__logstd__[i]))
        return z

    def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """
        loss = []
        for i in range(self.modals):
            pos_loss = -torch.log(
                self.decoder(z[i], pos_edge_index[i], sigmoid=True) + self.EPS).mean()

            if neg_edge_index is None:
                neg_edge_index = negative_sampling(pos_edge_index[i], z[i].size(0))
            neg_loss = -torch.log(1 -
                                  self.decoder(z[i], neg_edge_index, sigmoid=True) +
                                  self.EPS).mean()
            loss.append(pos_loss+neg_loss)
        return loss
    

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.

        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        loss = []
        for i in range(self.modals):
            mu = self.__mus__[i]
            logstd = self.__logstd__[i]
            loss.append(-0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)))
        return loss
    
    def consistLoss(self):
        loss = []
        for i in range(self.modals):
            loss.append(torch.sum((self.__mus__[i]-self.__mu__)**2))
        return loss


# In[ ]:





# In[8]:


# parameters
out_channels = 32
num_features = 3
warmUpEpoch = 5
maxEpoch = 200
modals = 3

# model
model = multiVGAE([VariationalGCNEncoder(num_features, out_channels) for i in range(modals)]).double()


# In[12]:


def train(model,histLoss,priority):
    model.train()
    optimizer.zero_grad()

    loss = [0]*model.modals
    for di in data:
        z = model.encode(di)
        loss1 = model.recon_loss(z, [d.edge_index for d in di])
        loss2 = model.kl_loss()
        loss3 = model.consistLoss()

        lossTmp = [loss1[i] + model.betaReg*loss2[i] + model.lambdaMu*loss3[i] for i in range(model.modals)]
        loss = [li+lti for li,lti in zip(loss,lossTmp)]


    if (epoch < warmUpEpoch)|(np.max(priority)<1/(model.modals+0.5)):
        lossFull = 0
        for i in range(model.modals):
            lossFull += loss[i]
        lossFull.backward()    
    else:
        loss[np.argmax(priority)].backward()
    optimizer.step()


    if epoch >= warmUpEpoch:
        curLoss = [i.detach().item() for i in loss]
        theta = [histLoss[i]-curLoss[i] for i in range(model.modals)]
        priority = [priority[i]/(10**-4+theta[i]) for i in range(model.modals)]
        priority = priority/np.sum(priority)
        histLoss = curLoss
    if histLoss is None:
        histLoss = [i.detach().item() for i in loss]

    print(f'epoch {epoch}, loss {sum(histLoss)}')
    
    return histLoss,priority


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
histLoss = None
priority = [1,1,1]
loader = DataLoader(data, batch_size=4, shuffle=True)

for epoch in range(maxEpoch):
    histLoss,priority = train(model,histLoss,priority)


# In[ ]:




