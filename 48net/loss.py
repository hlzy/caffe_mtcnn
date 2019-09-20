import matplotlib.pyplot as plt
import numpy as np

result = []
with open("tmp","r") as f:
    for eachline in f.readlines():
        loss = eachline.rstrip()
        result.append(float(loss))

plt.plot(range(len(result)),result)
#plt.xlabel("FPR")
#plt.ylabel("TPR")
#plt.text(0,0,"auc:%f" % auc)
#print(auc)
plt.savefig("./roi_loss.jpg")
