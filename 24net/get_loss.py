import matplotlib.pyplot as plt
import numpy as np

result = []
with open("m_loss.txt","r") as f:
    i = 0
    for eachline in f.readlines():
       i += 1
       prob,loss_ = eachline.rstrip().split()
       result.append([i,float(loss_)])
 
l_np = np.array(result)
count = l_np.shape[0]
plt.plot(l_np[:,0] * 500,l_np[:,1])
plt.savefig("./loss.jpg")
