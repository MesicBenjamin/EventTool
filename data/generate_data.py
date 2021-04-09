import numpy as np

# number of points
N = int(1e5)
f = 1./1e3

print('timestamp,x,y,p')

for i in range(N):

    x = round(50*np.sin(f*i) + 10*np.sin(10*i)*np.random.uniform() + 100)
    y = round(50*np.cos(f*i) + 10*np.sin(10*i)*np.random.uniform() + 100)
    t = i #np.random.choice([i, i+1, i+2])
    p = round(np.random.uniform())

    print(','.join(map(str, [t, x, y, p])))

    