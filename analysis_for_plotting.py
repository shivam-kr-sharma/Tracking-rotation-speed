import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import lombscargle
from scipy.fft import fft, fftfreq


f1 = open("sample2.txt","r")

lines = f1.readlines()
f1.close()

time = []
x = []
y = []
theta_x = []
theta_y = []
for i in range(1,len(lines)):
    temp = lines[i].split()
    time_value = float(temp[0])
    x_value = float(temp[1])
    y_value = float(temp[2])
    theta_x_value = math.atan(x_value*(14.5/640)*(1/45))
    theta_y_value = math.atan(y_value*(14.5/480)*(1/45))
    time.append(time_value)
    x.append(x_value)
    y.append(y_value)
    theta_x.append(theta_x_value)
    theta_y.append(theta_y_value)
    

newtime=[]
newx=[]
newy=[]
newtheta_x=[]
newtheta_y=[]
for i in range(1,len(time)):
    newtime.append(time[i])
    newx.append(x[i])
    newy.append(y[i])
    newtheta_x.append(theta_x[i])
    newtheta_y.append(theta_y[i])

# plt.plot(newtime,newtheta_x,  label =r"$ \theta_x$(in rad)")
# plt.plot(newtime,newtheta_y, label =r"$\theta_y$(in rad)")
# plt.xlabel("Time (s)")
# plt.ylabel(r"$\theta$ (rad)")
# plt.legend()


# plt.show()




## Fourier Transform processing

time = np.array(newtime)
theta_x = np.array(newtheta_x)
theta_y = np.array(newtheta_y)


f_min = 1/(time[-1]-time[0])
dt = np.diff(time)
f_max = 1/(2*np.median(dt))

frequencies = np.linspace(f_min,f_max,1000)
w = 2*np.pi*frequencies

pgram = lombscargle(time, theta_x, w)


plt.plot(frequencies, pgram)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Lomb-Scargle Periodogram")
plt.grid(True)
plt.show()


