from genetic1_1 import *
from genetic2 import *
from matplotlib import pyplot as plt

plt.plot(ls_gen1_1, ls_avgfit1_1)
plt.plot(ls_gen2, ls_avgfit2)
plt.xlabel('Generation')
plt.ylabel('Average of fitness value')
plt.legend(['Shuffle', 'Swap & Shuffle'])
plt.title('Comparison of two different strategies')
plt.show()