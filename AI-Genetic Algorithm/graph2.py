from genetic1_1 import *
from genetic1_2 import *
from matplotlib import pyplot as plt


plt.plot(ls_gen1_1, ls_avgfit1_1)
plt.plot(ls_gen1_2, ls_avgfit1_2)
plt.xlabel('Generation')
plt.ylabel('Average of fitness value')
plt.legend(['0.5%', '60%'])
plt.title('Comparison of two different parameters(probability)')
plt.show()

