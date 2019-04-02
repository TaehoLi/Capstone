import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
fig = plt.figure()
ax = fig.add_subplot(111)
fig.show()

i = 0
x, y = [], []
while True:
    x.append(i)
    y.append(psutil.cpu_percent())
    
    ax.plot(x, y, color='b')
    
    fig.canvas.draw()
    
    time.sleep(0.1)
    i += 1
"""
plt.ion()
for i in range(3000):
    plt.plot(np.random.rand(10))
    plt.show()
#안됨 뭔가 gui상에서 충돌이 일어나는듯
"""
