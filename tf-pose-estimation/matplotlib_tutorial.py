import numpy as np
import matplotlib.pyplot as plt
#matplotlib auto
#from datetime import datetime

"""
# data plotting
lstX = []
lstY = []
threshold = 0.5
    
plt.ion()
fig = plt.figure(num='real-time plotting')
sf = fig.add_subplot(111)
plt.title('stream data')
plt.xticks([0, 1500000, 3000000, 4500000, 6000000])
plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
line1, = sf.plot([0, 6000000], [0,1], 'b-')

result = [0.3,0.4,0.5,0.6,0.7,0.8]


while True:
    c = datetime.now()
    d = c.strftime('%S%f')
    d = np.float32(d) / 10.0

    lstX.append(d)
    lstY.append(result)

    if d > 5900000: # 1분마다 플롯 초기화
        fig.clf()
        sf = fig.add_subplot(111)
        plt.title('stream data')
        plt.xticks([0, 1500000, 3000000, 4500000, 6000000])
        plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2)
        lstX=[]
        lstY=[]
        line1, = sf.plot([0, 6000000], [0,1], 'b-')
        line1.set_data(lstX, lstY)
    else:
        line1.set_data(lstX, lstY)
        plt.show()
"""
'''
plt.ion()
plt.plot([1.6, 2.7])
plt.title("interactive test")
plt.xlabel("index")
ax = plt.gca()
ax.plot([3.1,2.2])
plt.draw()
'''

plt.ion()
for i in range(3000):
    plt.plot(np.random.rand(10))
    plt.draw()
#안됨 뭔가 gui상에서 충돌이 일어나는듯
