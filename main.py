import numpy as np

'''
Zero-Sum game
Players: 2
State space: GridWorld 4x4
    Some positions are blocked
    x = [px1,py2,px2,py2]
    px1,py1: position of player 1
    px2,py2: position of player 2
Action space:
    u = [ux1,uy1,ux2,uy2]
    ux1,uy1: action of player 1
    ux2,uy2: action of player 2
'''

# State space
Gmax = 3
Gmin = 0
Gblocked = []

# Number of states
Nx = (Gmax - Gmin + 1)**4 - len(Gblocked)**2

# State transition function
def next_state(x,u):
    return x + u

# Possible policies depending on the player
def g(player):
    policies = []
    # Possible combinations of policies depending on the player
    if player == 1:
        policies = [np.array([i,j,0,0]) for i in range(-1,2) for j in range(-1,2)]
    if player == 2:
        policies = [np.array([0,0,i,j]) for i in range(-1,2) for j in range(-1,2)]
    return policies

def states():
    # Table of possible states
    states = np.zeros((Nx,4))
    states[0,:] = [0,0,0,0]
    idx = 1
    for i in range(Gmin,Gmax+1):
        for j in range(Gmin,Gmax+1):
            for k in range(Gmin,Gmax+1):
                for l in range(Gmin,Gmax+1):
                    if [i,j] not in Gblocked and [k,l] not in Gblocked:
                        states[idx,:] = [i,j,k,l]
                        idx += 1
    return states
    
def isvalid_action(u,x):
    x_new = next_state(x,u)
    # check if the new state is valid
    if (x_new >= Gmin).any() and (x_new <= Gmax).any() and (x_new[0:2] not in Gblocked) and (x_new[2:4] not in Gblocked):
        return True
    else:
        return False
    
def instantaneous_cost(x):
    np.norm(x[0:2]-x[2:4])
    return 

# Initialize the table
T = 4 # Time horizon
V = np.zeros((T+1, Nx))
RewardT = 100
V[T,:] 
V[T,21] = 0 # The final state has cost 0

# Initialize the policy table
U_opt = np.zeros((T,Nx,4))

# Backward dynamic programming
for t in range(T-1,-1,-1):
    for i in range(Nx):
        x = states[i,:]
        policies = g(x)
        costs = []
        for policy in policies:
            u = policy
            if isvalid_action(u,x):
                x_next = next_state(x,u)
                costs.append(instantaneous_cost(x,u) + V[t+1,np.where((states == x_next).all(axis=1))[0][0]])
        V[t,i] = min(costs)
        U_opt[t,i,:] = policies[np.argmin(costs)]


'''
V = -np.ones((T+1,Nx)) # allocate space for value functions
Uopt = -np.ones((T,Nx),dtype=np.uint32) # allocate space for optimal actions
# initialize appropriately the terminal costs

V[T,:] = ...

for t in range(T, 0, -1):
    V[t-1,:], Uopt[t-1,:] = one_step_bdp(Nx,Nu,V[t,:])

# Trace the state with the optimal policy
x = ... # put here the initial state
xd = decode_state(x) # presented in a human-readable form
print("xd= ",xd)

for t in range(T):
    u = Uopt[t, x-1] # optimal action for state x
    ud = decode_action(u) # present it in human-readable form
    print("ud= ",ud)
    xn = next_state(x, u)
    xnd = decode_state(xn)
    print("xd= ",xnd)
    x = xn

#Please define the functions
#next_state(x,u),
#isvalid_action(u,x),
#instantaneous_cost(x,u),
#decode_state(x), and
#decode_action(u) according to your requirements.

'''