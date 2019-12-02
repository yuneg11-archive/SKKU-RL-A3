import sys
import matplotlib.pyplot as plt
import gym
from dqn import DQN

STEPS = 1000
LOG = False
POSTFIX = ""

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

env = gym.make('MountainCar-v0')

defaults = None
doubles = None
multisteps = None
pers = None

if default:
    env.reset()
    dqn = DQN(env, double_q=False, per=False, multistep=False)
    defaults = dqn.learn(STEPS)
    del dqn
if double:
    env.reset()
    dqn = DQN(env, double_q=True, per=False, multistep=False)
    doubles = dqn.learn(STEPS)
    del dqn
if multistep:
    env.reset()
    dqn = DQN(env, double_q=False, per=False, multistep=True)
    multisteps = dqn.learn(STEPS)
    del dqn
if per:
    env.reset()
    dqn = DQN(env, double_q=False, per=True, multistep=False)
    pers = dqn.learn(STEPS)
    del dqn

print("Reinforcement Learning Finish")
print("Draw graph ... ")

x = [i for i in range(STEPS)]

if default:
    if LOG:
        log = open("defaults" + POSTFIX + ".txt", "w")
        log.write("\n".join([str(avg) for avg in defaults]))
        log.close()
    plt.plot(x, defaults, label='DQN')
if double:
    if LOG:
        log = open("doubles" + POSTFIX + ".txt", "w")
        log.write("\n".join([str(avg) for avg in doubles]))
        log.close()
    plt.plot(x, doubles, label='Double')
if multistep:
    if LOG:
        log = open("multisteps" + POSTFIX + ".txt", "w")
        log.write("\n".join([str(avg) for avg in multisteps]))
        log.close()
    plt.plot(x, multisteps, label='Multistep')
if per:
    if LOG:
        log = open("pers" + POSTFIX + ".txt", "w")
        log.write("\n".join([str(avg) for avg in pers]))
        log.close()
    plt.plot(x, pers, label='PER')

plt.legend()
fig = plt.gcf()
plt.savefig("result" + POSTFIX + ".png")
plt.show()
