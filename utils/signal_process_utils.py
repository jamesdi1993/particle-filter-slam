from scipy import signal

def low_pass_imu(signals):
  b, a = signal.butter(1, 0.05) # First order filter, at 10Hz
  zi = signal.lfilter_zi(b, a)
  z, _ = signal.lfilter(b, a, signals, zi=zi * signals[0])
  return z
