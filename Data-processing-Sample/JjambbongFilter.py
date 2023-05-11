import numpy as np
import matplotlib.pyplot as plt
from LowpassFilter import *
from KalmanFilter import *

np.random.seed(0)

def get_volt():
    """Measure voltage."""
    v = np.random.normal(0, 2)   # v: measurement noise.
    volt_true = 14.4             # volt_true: True voltage [V].
    z_volt_meas = volt_true + v  # z_volt_meas: Measured Voltage [V] (observable).
    return z_volt_meas

# Input parameters.
time_end = 20
dt = 0.01
x_0 = 12  # 14 for book.
P_0 = 6

lpf = LowPassFilter(cutoff_freq=10, ts=0.01)
kalf = KalmanFilter(x_0=x_0, P_0=P_0)
lpkalf = KalmanFilter(x_0=x_0, P_0=P_0)

time = np.arange(0, time_end, dt)
n_samples = len(time)

volt_meas_save = np.zeros(n_samples)
volt_lpf_save = np.zeros(n_samples)
volt_esti_save = np.zeros(n_samples)
volt_lpesti_save = np.zeros(n_samples)

x_esti = 0
x_lpf = 0

for i in range(n_samples):
    z_meas = get_volt()

    x_lpf = lpf.filter(z_meas)
    x_esti = kalf.filter(z_meas)
    x_lpesti = lpkalf.filter(x_lpf)

    volt_meas_save[i] = z_meas
    volt_lpf_save[i] = x_lpf
    volt_esti_save[i] = x_esti
    volt_lpesti_save[i] = x_lpesti


plt.plot(time, volt_meas_save, 'r*--', label='Measurements')
plt.plot(time, volt_lpf_save, 'gv--', label='Lowpass Filter')
plt.plot(time, volt_esti_save, 'bo-', label='Kalman Filter')
plt.plot(time, volt_lpesti_save, 'cx-', label='LPK Filter')

plt.legend(loc='upper left')
plt.title('Measurements v.s. Estimation (Kalman Filter)')
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [V]')
plt.show()
#plt.savefig('png/simple_kalman_filter.png')