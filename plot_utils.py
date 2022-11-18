import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

def plot_normal(ax, mean, std, min_, max_):
  xrange = np.linspace(min_, max_, 50)
  log_p = -0.5*((xrange - mean)/std)**2  - np.log(std) - 0.5*np.log(2*np.pi)
  ax.plot(xrange, np.exp(log_p), c = 'r')

def plot_hist_normal(ax,v,label):
  ax.hist(v, bins=40,density=True,color="#555555");
  std = np.std(v)
  mean = np.mean(v)
  min_,max_ = v.min(), v.max()
  plot_normal(ax, mean, std, min_, max_)
  ax.set_xlabel(label)
  ax.set_ylabel("freq")

def draw_ellipse(pair,ax,r = 1,color='r'):
    # Covariance Recovered from Samples
    t = np.linspace(0, 2*np.pi, 100)
    xy_ellipse = np.vstack([pair.xscale*r*np.cos(t),pair.yscale*r*np.sin(t)]).T
    xy_ellipse = np.matmul(pair.rot, xy_ellipse.T).T
    ax.plot(pair.mean[0]+xy_ellipse[:,0] ,pair.mean[1]+xy_ellipse[:,1], c=color)

def plot_scatter_normal(ax,pair):
    ax.scatter(pair.v1,pair.v2,c="#555555")
    for t in np.linspace(0,1,5):
      pair.draw_ellipse(ax,r = scipy.stats.chi2(df=2).isf(t))
    ax.axis('equal')
