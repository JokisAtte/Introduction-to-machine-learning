import matplotlib.pyplot as plt
import numpy as np

# Linear solver
def my_linfit(x,y):
    a = 0
    b = 0
    model = np.polyfit(x,y,1)
    a,b = model[0], model[1]
    return a,b

# Main
x = np.random.uniform(-2,5,10)
y = np.random.uniform(0,3,10)
a,b = my_linfit(x,y)
plt.plot(x,y,'kx')
xp = np.arange(-2,5,0.1)
plt.plot(xp,a*xp+b,'r-')
print(f"My fit: a={a} and b={b}") #this line was broken in the example. it said a={b}
plt.show()