import serial


class MotionController():
	"""
	Used to control a CNC-like machine running Marlin firmware,
	through a serial port communication
	"""
	def __init__(self, port="/dev/ttyUSB0",baud=115200, disable_temp_report = True, *args, **kwargs) -> None:
		"""
		Creates the motion controller from port and baudrate
		
		:param port: serial port to connect to
		:param baud: baudrate to use
		"""
		self.serial=serial.Serial(port, baud)
		self.open=False

		self.disable_temp = disable_temp_report
	   
	def __enter__(self, *args, **kwargs):
		self.open=True
		self.serial.__enter__()

		if self.disable_temp:
			self.send_command("M155 S0")

	def __exit__(self, *args, **kwargs):
		self.open=False
		self.serial.__exit__(*args,**kwargs)
	   
	def send(self, command):
		"""
		Send `command` to serial followed by \r\n

		:param command: string containing the command to be sent
		"""
		if not self.open:
			raise Exception("Run this inside a with!")

		self.serial.write((command+"\r\n").encode())

	def read(self):
		"""
		Read and output from serial and return a `strip`ped string 
		"""
		if not self.open:
			raise Exception("Run this inside a with!")

		return self.serial.readline().decode().strip()

	def send_command(self, command, wait_ok=True):
		"""
		Send the command and wait for a response.

		Possibly the command by receiving an "ok\r\n" from serial.
		If `wait_ok` is specified, raise an exception if the response is "unkown command"

		:param command: string containing G-code command to be sent
		:param wait_ok: read lines from serial until the return is "ok\r\n". 
						Return the previous line, before the OK
		"""
		self.send(command)

		if not wait_ok:
			return self.read()

		out=None
		ret = self.read()
		while ret != "ok":
			out = ret
			ret = self.read()
		
		if out and "echo:Unknown command:" in out:
			raise Exception("Unkown Command!")

		return out

	def move_to(self, x=None, y=None, z=None):
		"""
		Blocks execution while the machine moves to (x,y,z)
		"""
		x_str = "" if x is None else f" X{x}" 
		y_str = "" if y is None else f" Y{y}" 
		z_str = "" if z is None else f" Z{z}" 
		command = f"G0{x_str}{y_str}{z_str}"
		self.send_command(command)
		self.send_command("M114")