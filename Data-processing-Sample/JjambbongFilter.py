import numpy as np
import matplotlib.pyplot as plt
from LowpassFilter import *
from KalmanFilter import *


class Sensor:

    # volt_true: True voltage [V].
    volt_true = 2.9
    z_volt_meas = 0
    ts = 0

    def __init__(self):
        np.random.seed(0)
        pass

    """Measure voltage."""
    def get_volt(self):
        v = np.random.normal(-0.7, 0.7)  # v: measurement noise.

        self.volt_true = self.volt_true - (self.ts * 0.001)

        self.z_volt_meas = self.volt_true + v  # z_volt_meas: Measured Voltage [V] (observable).

        self.ts = self.ts + 1
        return self.z_volt_meas


# Input parameters.
time_end = 10
dt = 0.15

x_0 = 2.9 #초기값
P_0 = 2

sensor = Sensor()

lpf = LowPassFilter(init=x_0, cutoff_freq=5, ts=dt * 0.1)
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

for step in range(n_samples):

    z_meas = sensor.get_volt()

    x_lpf = lpf.filter(z_meas)
    x_esti = kalf.filter(z_meas)
    x_lpesti = lpkalf.filter(x_lpf)

    volt_meas_save[step] = z_meas
    volt_lpf_save[step] = x_lpf
    volt_esti_save[step] = x_esti
    volt_lpesti_save[step] = x_lpesti


plt.plot(time, volt_meas_save, 'r.--', label='Measurements')
plt.plot(time, volt_lpf_save, 'gv--', label='Lowpass Filter')
plt.plot(time, volt_esti_save, 'cx-', label='Kalman Filter')
plt.plot(time, volt_lpesti_save, 'bo-', label='LPK Filter')

plt.legend(loc='upper left')
plt.title('Measurements v.s. Estimation (Kalman Filter)')
plt.xlabel('Time [sec]')
plt.ylabel('Voltage [V]')
plt.show()
#plt.savefig('png/simple_kalman_filter.png')