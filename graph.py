import sys
import matplotlib.pyplot as plt

STEPS = 1000

default, double, multistep, per = False, False, False, False

if len(sys.argv) == 1:
    print("There is no argument, please input")

for i in range(1, len(sys.argv)):
    if sys.argv[i] == "default":
        default = True
    elif sys.argv[i] == "double":
        double = True
    elif sys.argv[i] == "multistep":
        multistep = True
    elif sys.argv[i] == "per":
        per = True

x = [i for i in range(STEPS)]

if default:
    log = open("defaults.txt", "r")
    data = log.readlines()
    defaults = [float(avg) for avg in data]
    log.close()
    plt.plot(x, defaults, label='DQN')
if double:
    log = open("doubles.txt", "r")
    data = log.readlines()
    doubles = [float(avg) for avg in data]
    log.close()
    plt.plot(x, doubles, label='Double')
if multistep:
    log = open("multisteps.txt", "r")
    data = log.readlines()
    multisteps = [float(avg) for avg in data]
    log.close()
    plt.plot(x, multisteps, label='Multistep')
if per:
    log = open("pers.txt", "r")
    data = log.readlines()
    pers = [float(avg) for avg in data]
    log.close()
    plt.plot(x, pers, label='PER')

plt.legend()
fig = plt.gcf()
plt.savefig("result.png")
plt.show()
