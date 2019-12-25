import numpy as np
import funs as f

lambda_l = [i / 10 for i in range(0, 11, 1)]

N = int(np.log(1 - 0.95) / np.log(1 - 0.01 / (np.pi - 0)))

X = np.linspace(start=0., stop=np.pi, num=101)
Y = np.sin(X) + 0.5

Y_with_noise = f.make_a_noise(Y)

f.will_work(X, Y, Y_with_noise, 3, N, lambda_l)
f.will_work(X, Y, Y_with_noise, 5, N, lambda_l)
