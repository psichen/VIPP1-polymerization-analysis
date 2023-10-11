import numpy as np
from matplotlib import pyplot as plt

def get_angle(x0,y0, x, y):
    a = np.arctan2(y-y0, x-x0)
    a = np.delete(a, (y-y0==0)*(x-x0==0))
    return a

def get_dist(x0,y0,x,y):
    d = np.sqrt((x-x0)**2 + (y-y0)**2)
    return d

theta_max = 6*np.pi
theta_step = .01
theta = np.linspace(-12*np.pi, theta_max, int(theta_max/theta_step))
r = 45.5*np.exp(.041*theta)
x = r*np.cos(theta)
y = r*np.sin(theta)

# inwards
x0 = x[-1]
y0 = y[-1]

a = get_angle(x0,y0,x,y)
a[a<0] += 2*np.pi

f1 = plt.figure()
plt.plot(x, y, '-')
plt.plot(x0,y0,'*')

a_inwards = np.array([130, 155, 180, 205, 230])/180*np.pi
d_inwards = []
for r in a_inwards:
    i = np.where(np.abs(a-r)<1e-1)[0]
    d = get_dist(x0,y0,x[i],y[i])
    d_max = np.max(d)
    d_inwards.append(d_max)

    plt.plot([x0,x[i][d==d_max][0]], [y0,y[i][d==d_max][0]], '--', color='gray')

f2 = plt.figure()
plt.plot(a/np.pi*180, '.')
plt.xlabel('index')
plt.ylabel('angle (degre)')

# outwards
x0 = x[0]
y0 = y[0]

a = get_angle(x0,y0,x,y)
a[a<0] += 2*np.pi

f3 = plt.figure()
plt.plot(x, y, '-')
plt.plot(x0, y0,'*')

a_outwards = np.array([36, 108, 180, 252, 324])/180*np.pi
d_outwards = []
for r in a_outwards:
    i = np.where(np.abs(a-r)<1e-1)[0]
    d = get_dist(x0,y0,x[i],y[i])
    d_max = np.max(d)
    d_outwards.append(d_max)

    plt.plot([x0,x[i][d==d_max][0]], [y0,y[i][d==d_max][0]], '--', color='gray')

f4 = plt.figure()
plt.plot(a/np.pi*180, '.')
plt.xlabel('index')
plt.ylabel('angle (degre)')

f5 = plt.figure()
plt.plot(a_outwards/np.pi*180, d_outwards, '*')
plt.plot(a_inwards/np.pi*180, d_inwards, '*')
plt.xlabel('angle (degree)')
plt.ylabel('distance (nm)')
plt.xlim(0,359)
plt.ylim(0,300)

plt.show()

# f1.savefig('./spiral_inwards.pdf', transparent=True)
# f3.savefig('./spiral_outwards.pdf', transparent=True)

# np.save('./angle_inwards', a_inwards)
# np.save('./dist_inwards', d_inwards)
# np.save('./angle_outwards', a_outwards)
# np.save('./dist_outwards', d_outwards)
