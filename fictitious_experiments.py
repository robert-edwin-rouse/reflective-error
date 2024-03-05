'''
Fictitious experiments demonstrating the use of reflective error based on
simple Gaussian datasets used in the paper "Reflective Error: A Metric for
Assessing Predictive Performance at Extremes".

'''

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from apollo import metrics as me
from apollo import mechanics as ma


np.random.seed(42)
ma.textstyle()


### Gaussian distribution class
class Gaussian_Data_Model():
    def __init__(self, name, x, mu, sigma):
        '''
        Initialises an instance of a Gaussian model where parameters are fitted
        to a dataset resampled from a given dataset in order to conduct
        comparative analysis.

        Parameters
        ----------
        name : String
            Name of model or scenario.
        x : Array
            Initial array of values.
        mu : Float
            Gaussian distribution mean.
        sigma : Float
            Gaussian distribution standard deviation.

        Returns
        -------
        None.
        '''
        self.name = str(name)
        self.x = x
        self.mu = mu
        self.sigma = sigma
        self.sample = np.sort(np.random.normal(self.mu, self.sigma, len(self.x)))
        self.mu_o, self.sigma_o = sp.stats.norm.fit(self.sample)
        self.kappa = 1/(self.sigma_o * np.sqrt(2*np.pi))

    def u_of_x(self, x):
        '''
        Calculates the probabilities for corresponding x values.

        Parameters
        ----------
        x : Array
            Input array.

        Returns
        -------
        u : Array
            Probability density array.
        '''
        u = 1/(self.sigma_o*np.sqrt(2*np.pi)) * \
                    np.exp(-(x-self.mu_o)**2/(2 * self.sigma_o**2))
        return u
    
    def psi(self, x):
        '''
        Calculates the reflective weighting, psi, for x values.

        Parameters
        ----------
        x : Array
            Input array.

        Returns
        -------
        psi : Array
            Reflective weighting array.
        '''
        psi = (-1)*(self.u_of_x(x))/self.kappa + 1
        return psi

    def histPlot(self, s=6):
        '''
        Generates a histogram for an input dataset with the pdf shown on the
        left axis and the reflective weighting shown on the right axis.

        Parameters
        ----------
        s : Float, optional
            Size for the plot. The default is 6.

        Returns
        -------
        The histogram plot outputted to the viewport.
        '''
        fig, ax1 = plt.subplots(figsize=(s, s))
        h = np.random.normal(self.mu, self.sigma, len(self.x))
        count, bins, ignored = plt.hist(h, int(np.round(len(self.x)/25)), 
                                        density=True,rwidth=0.8,
                                        color='darkseagreen', alpha=0.6)
        ax1.plot(self.x, self.u_of_x(self.x), 'cadetblue', lw=4, 
                 ls='-', label='U(x)')
        ax1.set_xlim(self.x.min(), self.x.max())
        ax1.set_xlabel('x')
        ax1.set_ylim(0, np.round(self.u_of_x(self.x).max(), 2)/0.625)
        ax1.set_ylabel('U'+' (x)')
        ax1.grid(c='black', ls='dotted', lw=0.5)
        ax2 = ax1.twinx()
        ax2.plot(self.x, self.psi(self.x), 'orange', lw=4, ls='-.',
                 label=r'$\Psi$(x)')
        ax2.set_ylim(0, 1.5)
        ax2.set_ylabel(r'$\Psi$(x)')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
        plt.show()
    
    def report_error(self, scenario, data):
        '''
        Generates error metrics between the class data and another set of data,
        including root mean squared error, r^2, and reflective error.

        Parameters
        ----------
        other : Array
            The second set of data being compared against the class sample data.
        scenario : String
            The name of the second dataset being compared.

        Returns
        -------
        rmse : Float
            Root mean squared error.
        r2 : Float
            R2 score.
        re : Float
            Reflective Error score.
        '''
        rmse = me.RMSE(self.sample, data)
        r2 = me.R2(self.sample, data)
        re = me.RE(self.sample, data, self.psi(self.sample))
        print('- - - - - - - - - - - - - - -')
        print(self.name + ' vs. ' + scenario)
        print('- - - - -')
        print('RMSE: ' + str(rmse))
        print('R2: ' + str(r2))
        print('RE: ' + str(re))
        print('- - - - - - - - - - - - - - -')
        return rmse, r2, re
    
    def versus_plot(self, data, s=6):
        '''
        Generates a scatter plot of the class sample data plotted against a
        second dataset.

        Parameters
        ----------
        other : Array
            The second set of data being compared against the class sample data.
        s : Float, optional
            Size for the plot. The default is 6.

        Returns
        -------
        The scatter plot outputted to the viewport.
        '''
        xyline = np.linspace(self.x.min(), self.x.max())
        fig, ax = plt.subplots(figsize=(s, s))
        ax.scatter(data, self.sample, s=(s*2), edgecolors='darkseagreen',
                   lw=1.6, facecolors='none', marker = 'o')
        ax.plot(xyline, xyline, c='black', lw=1, alpha=0.5)
        ax.set_xlim(self.x.min(), self.x.max())
        ax.set_xlabel('Predictions')
        ax.set_ylim(self.x.min(), self.x.max())
        ax.set_ylabel('Observations')
        ax.grid(c='black', ls='dotted', lw=0.5)
        plt.show()


### Example relevance function for a Gaussian distribution
x1 = np.linspace(-1, 1, 1000)
g1 = Gaussian_Data_Model('Example', x1, 0, 0.25)
g1.histPlot()


### Setup observations
mu, sigma, n = 3.5, 0.75, 500
x2 = np.linspace(1, 6, n)
g2 = Gaussian_Data_Model('Model', x2, mu, sigma)
noise = np.random.normal(0, 0.2, len(x2))


### Unperturbed scenario
base_scenario = 'Unperturbed Data'
base_data = g2.sample + noise
_, _, _ = g2.report_error(base_scenario, base_data)
g2.versus_plot(base_data)


### Extreme perturbation scenario
extreme_scenario = 'Extremity Perturbed Data'
extreme_data = g2.sample + noise
extreme_data[:int(n/10)] += np.random.normal(0, 1.0, 50)
extreme_data[-int(n/10):] += np.random.normal(0, 1.0, 50)
_, _, _ = g2.report_error(extreme_scenario, extreme_data)
g2.versus_plot(extreme_data)


### Central perturbation scenario
central_scenario = 'Centrally Perturbed Data'
central_data = g2.sample + noise
central_data[int(n/2-n/10):int(n/2+n/10)] += np.random.normal(0, 1.0, 100)
_, _, _ = g2.report_error(central_scenario, central_data)
g2.versus_plot(central_data)
