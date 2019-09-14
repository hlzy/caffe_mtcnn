import matplotlib.pyplot as plt
import numpy as np

result = []
with open("roc_file_24.txt","r") as f:
    for eachline in f.readlines():
        label,prob = eachline.rstrip().split()
        result.append([float(prob),int(label)])

result = sorted(result)
r_np  = np.array(result)
count = r_np.shape[0]
N_0 = np.sum(r_np[:,1] == 0)
N_1 = np.sum(r_np[:,1] == 1)
x = []
y = []
last_TPR = 1
last_FPR = 1
auc = 0 
for i in range(count):
   TP = np.sum(r_np[i:,1] == 1)
   FP = np.sum(r_np[i:,1] == 0)
   TPR = float(TP) / N_1
   FPR = float(FP) / N_0
   x.append(FPR)
   y.append(TPR)
   auc += float(last_TPR  + TPR) * (last_FPR - FPR)  / 2
   last_TPR = TPR
   last_FPR = FPR
   

plt.plot(x,y)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.text(0,0,"auc:%f" % auc)
print(auc)
plt.savefig("./roc.jpg")
