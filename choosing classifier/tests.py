import tensorflow as tf
tf._logging.set_verbosity(tf._logging.ERROR)
import numpy as np

celcius= np.array([-40,-10,0,8,15,22,38], dtype=float)
fahrenheit=np.array([-40,-14,32,46,59,72,100.4], dtype=float)
for i,c in enumerate(celcius):
  print("{} degree celcius = {} degree fahrenheit".format(c,fahrenheit[i]))