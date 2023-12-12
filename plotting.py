import math
import matplotlib.pyplot as plt

max_steps = 50000
limit_steps = 1000
warm_up_steps = max_steps
warm_up_1st_phase_steps = 2000
warm_up_2nd_phase_steps = 5000
warm_up_3rd_phase_steps = 15000

lr_list = []
for global_step in range(max_steps):
    if global_step < warm_up_1st_phase_steps:
        lr_list += [1.]
    else:
        lr_list += [0.5 * (1 + math.cos(math.pi * (global_step / limit_steps)))]
x = list(range(max_steps))

plt.plot(x, lr_list, color='r', linewidth=2)
plt.show()
