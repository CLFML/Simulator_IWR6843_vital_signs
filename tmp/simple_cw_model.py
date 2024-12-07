from matplotlib import pyplot as plt
from numpy import repeat, roll, pi, linspace, cos, sin, fft, log10, arange, arctan2, unwrap, random, sqrt

# radar constants
f_CW = 60e9 # [Hz] carrier frequency  
s = 50e-3 # [Hz/s] MHz/ns
c = 3e8 # [m/s] speed of light

# target constants
delta_x_breathing = .001 # [m] thorax displacement
f_breathing = 0.1 # [Hz] breathing rate
d = 1 # [m] distance
# reflection coefficient, radar cross-section (sigma), frequency dependent
delta_x_heartbeat = .0001 # [m] thorax displacement, much smaller amplitude than breathing
f_heartbeat = 1.2  # [Hz] typical heartbeat frequency

delta_x_motion_artifact = 0.00005 # [m] random body movement

# baseband signal constants
f_s = 10 # [Sps]
n_phi = pi/8 # [rad] phase noise amplitude

# signal model
N = 8 * 1024 # number of samples
T_s = 1 / f_s
t = arange(0, N * T_s, T_s)
freq = linspace(0.0, f_s/2, N//2) # get freq axis

# f = f_CW + s*t # [Hz] frequency
# f = repeat(f, 2) # [Hz] frequency
# f = roll(f, 1000) # [Hz] frequency
x_breathing = delta_x_breathing * cos(2*pi*f_breathing*t) # [V] displacement signal
x_heart = 0.0001 * cos(2*pi*f_heartbeat*t)  # [m] 
x_motion_artifact = delta_x_motion_artifact * random.randn(N)  # Random body movement

x = x_breathing + x_heart + x_motion_artifact # Combined displacement
phi = n_phi * random.randn(N) # [rad] phase noise

lambda_CW = c/f_CW # [m] wavelength 
path_loss = (c/(4*pi*f_CW*d))**2  # Free space path loss
I = cos(4 * pi * d / lambda_CW + 4 * pi * x / lambda_CW + phi) # In-phase signal
Q = cos(4 * pi * d / lambda_CW + 4 * pi * x / lambda_CW + phi - pi/2) # Quadrature signal
s = sqrt(path_loss) * (I + 1j*Q)

P_i = fft.fft(I)
P_q = fft.fft(Q)
P = fft.fft(s)


fig, ax = plt.subplots()
ax.plot(t, I, 'b')
ax.plot(t, Q, 'r')
ax.plot(t, unwrap(arctan2(I, Q)), 'g')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Amplitude [V]')
ax.set_title('CW radar signal')
ax.legend(['I', 'Q', 'phase'])


fig1, ax1 = plt.subplots()
ax1.semilogx(freq, 20*log10(abs(P_i[1:N//2+1])), 'b')
ax1.semilogx(freq, 20*log10(abs(P_q[1:N//2+1])), 'r')
ax1.semilogx(freq, 20*log10(abs(P[1:N//2+1])), 'g')
ax1.set_xlabel('Frequency [Hz]')
ax1.set_ylabel('Amplitude [dB]')
ax1.grid()
ax1.set_title('CW radar signal')
ax1.legend(['I', 'Q', 'P'])
ax1.set_xlim([1e-2, f_s/2])

plt.show()