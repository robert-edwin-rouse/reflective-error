'''
Python script for reproducing the results of the reflective error streamflow
model.

@author: robert-edwin-rouse
'''

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scipy as sp
from torch.autograd import Variable
from apollo import streamflow as sf
from apollo import metrics as me
from apollo import mechanics as ma


### Set plotting style parameters
ma.textstyle()


### Set global model parameters
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### Import data as dataframe and remove unclean data rows
station = 53018
filename = str(str(station) + '_lumped.csv')
rf = pd.read_csv(filename)
rf['Date'] = pd.to_datetime(rf['Date'], format='%Y-%m-%d').dt.date


### Identify features (with either antecedent proxies or soil moisture levels)
days = 6
features = ['Rain'] + ['Rain-' + f'{d+1}' for d in range(days)] \
            + ['Temperature'] \
            + ['Temperature-' + f'{d+1}' for d in range(days)] \
            + ['Resultant Windspeed'] \
            + ['Resultant Windspeed-' + f'{d+1}' for d in range(days)] \
            + ['Humidity'] + ['Humidity-' + f'{d+1}' for d in range(days)] \
            + ['Rain_28_Mu','Rain_90_Mu','Rain_180_Mu',
               'Temperature_28_Mu','Temperature_90_Mu','Temperature_180_Mu']
targets = ['Flow']
xspace = ma.featurelocator(rf, features)
yspace = ma.featurelocator(rf, targets)


###Test/Train data split by years
yearlist = [2009+i for i in range(12)]
rftrain = rf[~pd.to_datetime(rf['Date']).dt.year.isin(yearlist)]


### Fit distribution to training set data & set reflective penalty
def lognorm_prob(x, sigma, loc, scale):
    '''
    Calculates the lognormal probability density function at x.

    Parameters
    ----------
    x : Float
        Input x.
    sigma : Float
        Standard deviation.
    loc : Float
        DESCRIPTION.
    scale : Float
        DESCRIPTION.

    Returns
    -------
    u_of_x : The value of the lognormal probability density function at x.
    '''
    u_of_x = (1/(sigma*(x-loc)/scale*((np.pi*2)**(1/2)))) \
        * np.exp(-(np.log((x-loc)/scale)**2)/(2*sigma**2)) / scale*2
    return u_of_x

flows = rftrain['Flow'].values
sigma, loc, scale = sp.stats.lognorm.fit(flows)
alpha = 1
beta = 2
kappa = np.exp(-2*(sigma**2)) * (loc*np.exp(2*(sigma**2))+scale)
relevance = lognorm_prob(kappa, sigma, loc, scale)
rf['U_of_y'] = lognorm_prob(rf['Flow'], sigma, loc, scale)
rf['Psi'] = me.RELossWeight(rf['U_of_y'], 1, 1, relevance)
rf['Relevance'] = me.RELossWeight(rf['U_of_y'], 1, 2, relevance)



### Normalise features using parameters cached from the training set
norm_cache = {}
for f in features:
    rftrain[f] = ma.normalise(rftrain, f, norm_cache, write_cache=True)
    rf[f] = ma.normalise(rf, f, norm_cache, write_cache=False)


### Convert dataframe subsets to arrays and then to PyTorch variables
rftrain = rf[~pd.to_datetime(rf['Date']).dt.year.isin(yearlist)]
trnset = rftrain.to_numpy()
X = trnset[:,xspace].reshape(len(trnset), len(xspace)).astype(float)
Y = trnset[:,yspace].reshape(len(trnset), len(yspace)).astype(float)
G = trnset[:,rf.columns.get_loc('Relevance')].reshape(len(trnset), 1).astype(float)
x = torch.from_numpy(X).to(device)
y = torch.from_numpy(Y).to(device)
g = torch.from_numpy(G).to(device)
x, y = Variable(x), Variable(y)


### Define Neural Network structure and initialisation procedure
class AntecedentNET(nn.Module):
    def __init__(self, in_dim, out_dim):
        '''
        Neural streamflow model class instantiation.

        Parameters
        ----------
        in_dim : Integer
            DESCRIPTION.
        out_dim : Integer
            DESCRIPTION.

        Returns
        -------
        None.
        '''
        super(AntecedentNET, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear_layers = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            )
    
    def forward(self, z):
        '''
        Performs a forward pass through the network

        Parameters
        ----------
        z : Tensor
            Model input.

        Returns
        -------
        z : Tensor
            Model output.
        '''
        z = self.linear_layers(z)
        return z

def init_weights(m):
    '''
    Initialises weights in neural models.

    Parameters
    ----------
    m : Pytorch neural network model
        Neural model with weights to be initialised.

    Returns
    -------
    None.

    '''
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)


### Network initialisation
net = AntecedentNET(len(xspace), len(yspace))
net = nn.DataParallel(net)
net.apply(init_weights)


### Network training
net = net.train()
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.1)
loss_list = []
for i in range(5000):
    y_pred = net(x.float())
    loss = torch.abs(torch.mean(me.RELossFunc(y_pred, y.float(), g.float())))
    net.zero_grad()
    loss.backward()
    optimizer.step()
    loss_list.append(loss.data)
    if(i % 500 == 0):
        print('epoch {}, loss {}'.format(i, loss.data))


### Evaluate Network
net = net.eval()
fullset = rf.to_numpy()
Z = fullset[:,xspace].reshape(len(fullset), len(xspace)).astype(float)
z = torch.from_numpy(Z).to(device)
predicted = net(z.float()).data.cpu().numpy()
rf['Predicted'] = predicted
maxflow = 300
rftrain = rf[~pd.to_datetime(rf['Date']).dt.year.isin(yearlist)]
rfvalid = rf[pd.to_datetime(rf['Date']).dt.year.isin(yearlist[0:1])]
rftest = rf[pd.to_datetime(rf['Date']).dt.year.isin(yearlist[1:])]
for df in (rftrain, rfvalid, rftest):
    sf.scatter_plot(maxflow, df, 'Predicted', 'Flow')
    print('- - - - - - - - - - - - - - -')
    print('RMSE: ' + str(me.RMSE(df['Flow'], df['Predicted'])))
    print('R\N{SUPERSCRIPT TWO}: ' + str(me.R2(df['Flow'], df['Predicted'])))
    print('RE: ' + str(me.RE(df['Flow'], df['Predicted'], df['Psi'])))
sf.year_plot(maxflow, rf, 'Predicted', 'Flow', 2007)
sf.year_plot(maxflow, rf, 'Predicted', 'Flow', 2009)
sf.year_plot(maxflow, rf, 'Predicted', 'Flow', 2012)