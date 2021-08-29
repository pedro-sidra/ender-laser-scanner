from rpi_hardware_pwm import HardwarePWM
import time

time.sleep(2)

pwm = HardwarePWM(1, hz=60)
pwm.start(0) # full duty cycle

# good values
pwm.change_duty_cycle(5)
time.sleep(2)
pwm.change_duty_cycle(10)

pwm.stop()
