import pandas as pd 
import numpy as np 
from scipy.stats import rankdata 
from scipy.stats import multivariate_normal 

def get_rbc(cont, bin, n):
  ranks = rankdata(cont, method = 'average')

  y_1 = np.mean(ranks[bin == 1])
  y_0 = np.mean(ranks[bin == 0])
  rbc = 2 * (y_1 - y_0) / n
  return rbc


def rbcorr(cont, bin, boot_ci = True, boot_n = 10000):
  """
  computes the rank biserial correlation between two arrays cont and bin
  cont: array of numberic values 
  bin: array of binary values 
  The function considers 1 as the base group >> i.e. positive correlations mean larger values in gorup 1
  """
  if len(cont) != len(bin):
    raise ValueError("Arrays must be of equal length.")
  
  if set(bin) - set([0,1]):
    raise ValueError("bin must contain only binary values.")
  
  if not isinstance(boot_ci,(bool)):
    raise ValueError("boot_ci must be boolean")
  
  if not isinstance(boot_n, (int)):
    raise ValueError("boot_n must be an integer")

  if boot_n <= 0:
    raise ValueError("boot_n must be greater than 0")

  n = len(cont)
  rbc = get_rbc(cont,bin,n)

  # bootstrap 
  if boot_ci:
    boot = np.zeros(boot_n)

    for i in range(boot_n):
      id = np.random.choice(n,n)
      cont_i = cont[id]
      bin_i = bin[id]
      boot[i] = get_rbc(cont_i,bin_i,n)

    se = np.std(boot)
    ci = np.array([rbc - 1.96 * se, rbc + 1.96 * se])
    return {'rbc': rbc, 'ci': ci}
  
  return rbc


#making some data to check if everything works as expected 
rv = multivariate_normal(mean = [0,0], cov = [[1,-0.9],[-0.9,1]])
sample = rv.rvs(size = 1000)

delay = sample[:,0]
reorder = (sample[:,1] < delay).astype(int)

rbcorr(delay,reorder, True, 100)

print('mean for group 0:', np.mean(delay[reorder == 0]))
print('mean for group 1:', np.mean(delay[reorder == 1]))