import numpy as np
import scipy.stats
import sklearn.metrics
import itertools as it

@np.vectorize
def gauss_minus_log_p(x, mean, std):
  return 0.5*((x - mean)/std)**2   + np.log(std) + 0.5*np.log(2*np.pi)

def integral_gauss_minus_log_p(a,b, mean, std):
  return ((b - mean)**3-(a - mean)**3)/(6*std**2)   + (a-b)*(np.log(std) + 0.5*np.log(2*np.pi))

class SingleInfo:
  """ Contains various information-theoretic descriptors for a 1-dimennsional dataset"""

  def __init__(self, v, n_bins=10):
    self.v=v
    self.mean = np.mean(self.v)
    self.std  = np.std(self.v)
    
    # normal distributio that fits the data
    self.d = scipy.stats.norm(self.mean,self.std)
    
    # calculate a histogram and treat it as empirical probability distribution
    count, bins = np.histogram(self.v, bins=n_bins)
    self.p_intervals = count/len(self.v)
    self.intervals = list(zip(bins[:-1],bins[1:]))
  
  def empirical_entropy(self):
    out = 0 # find empirical_entropy by summing across intervals where p is constant
            # i.e. bins in the original histogram
    for p,interval in zip(self.p_intervals, self.intervals):
      a,b = interval
      if p != 0:
        out+=-p*np.log(p)
    return out

  def normal_entropy(self):
    return self.d.entropy()

  def normal_entropy_homemade(self):
    return 0.5*np.log(2*np.pi*self.std**2)+0.5

  def cross_to_normal(self):
    out = 0 # find cross_entropy by summing across intervals where p is constant
            # i.e. bins in the orignal histogram
    for p,interval in zip(self.p_intervals, self.intervals):
      a,b = interval
      if p != 0: 
        integral = self.d.cdf(b) - self.d.cdf(a)
        surprisal = -np.log(integral)
        out+=p*surprisal
    return out

  def kl_div_to_normal(self):
    return self.cross_to_normal()-self.empirical_entropy()




def recover_ellipse(covariance):
  diag, rot = np.linalg.eigh(covariance)
  xscale, yscale = np.sqrt(diag)
  return xscale,yscale,rot

def block_p_from_cdf(cdf, x_interval, y_interval):
  x0,x1 = x_interval
  y0,y1 = y_interval
  return cdf((x0, y0)) + cdf((x1,y1)) -cdf((x0, y1)) - cdf((x1,y0))

class PairInfo:

  def __init__(self, v1, v2,n_bins=50):

    self.v1 = v1
    self.v2 = v2

    self.part1 = SingleInfo(v1)
    self.part2 = SingleInfo(v2)

    self.xy = np.vstack([v1,v2]).T
    self.covariance = np.cov(self.xy.T)
    self.xscale,self.yscale,self.rot = recover_ellipse(self.covariance)
    self.mean = np.mean(self.xy,axis=0)
    self.distrib = scipy.stats.multivariate_normal(mean=self.mean, cov=self.covariance)

    # calculate a histogram and treat it as empirical probability distribution.
    count, xbins,ybins = np.histogram2d(v1,v2, bins=n_bins)
    self.p_intervals = count/len(self.xy.flatten())
    self.xintervals = list(zip(xbins[:-1],xbins[1:]))
    self.yintervals = list(zip(ybins[:-1],ybins[1:]))

  def draw_ellipse(self,ax,r = 1,color='r'):
    # Covariance Recovered from Samples
    t = np.linspace(0, 2*np.pi, 100)
    xy_ellipse = np.vstack([self.xscale*r*np.cos(t),self.yscale*r*np.sin(t)]).T
    xy_ellipse = np.matmul(self.rot, xy_ellipse.T).T
    ax.plot(self.mean[0]+xy_ellipse[:,0] ,self.mean[1]+xy_ellipse[:,1], c=color)

  def normal_entropy(self):
    return scipy.stats.multivariate_normal(self.covariance).entropy()
  
  def empirical_entropy(self):
    out = 0 
    for p,block in zip(self.p_intervals.flatten(), it.product(self.xintervals,self.yintervals)):
      xinterval,yinterval = block
      if p != 0:
        out+= -1*p*np.log(p)
    return out

  def cross_to_normal(self):
    out = 0 # find cross_entropy by summing across intervals where p is constant
            # i.e. bins in the original histogram
    for p,block in zip(self.p_intervals.flatten(), it.product(self.xintervals,self.yintervals)):
      xinterval,yinterval = block
      if p != 0:
        out+=p*-np.log(block_p_from_cdf(self.distrib.cdf, xinterval,yinterval))
    return out

  def individual_entropy(self):
    out = 0
    for p,block in zip(self.p_intervals.flatten(), it.product(self.xintervals,self.yintervals)):
      xinterval,yinterval = block
      if p != 0:
        out+=p*-np.log(block_p_from_cdf(self.distrib.cdf, xinterval,yinterval))
    return out

  def collect(self):

    temp = {}
    temp["individual_cross_to_normal"] = self.part1.cross_to_normal() + self.part2.cross_to_normal()
    temp["individual_entropy"] = self.part1.empirical_entropy() + self.part2.empirical_entropy()
    temp["cross_to_normal"] = self.cross_to_normal()
    temp["empirical_entropy"] = self.empirical_entropy()
    temp["xscale"] = self.xscale
    temp["yscale"] = self.yscale
    temp["theta"] = np.arctan2(self.rot[0,0],self.rot[0,1])
    return temp

