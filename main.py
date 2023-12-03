import numpy as np
from scipy.optimize import linprog
import animation as anim    # animation.py
import os

'''
Zero-Sum game
Players: 2
State space: GridWorld Gmax x Gmax
    Some positions are blocked
    x = [px1,py2,px2,py2]
    px1,py1: position of player 1
    px2,py2: position of player 2
Action space:
    u = [ux1,uy1,ux2,uy2]
    ux1,uy1: action of player 1
    ux2,uy2: action of player 2
'''

# State transition function
def next_state(x,u):
    return x + u

def finalValue(reward, Nx, Map):
    Gmin, Gmax, Gblocked = Map
    # Final value function
    V = np.zeros((Nx,))
    idx = 0
    for i in range(Gmin,Gmax+1):
        for j in range(Gmin,Gmax+1):
            for k in range(Gmin,Gmax+1):
                for l in range(Gmin,Gmax+1):
                    if [i,j] not in Gblocked and [k,l] not in Gblocked:
                        if i == k and j == l:
                            V[idx] = reward
                        else:
                            V[idx] = -reward
                        idx += 1
    return V

def stateSet(Nx, Map):
    Gmin, Gmax, Gblocked = Map
    # Table of possible states
    states = np.zeros((Nx,4))
    idx = 0
    for i in range(Gmin,Gmax+1):
        for j in range(Gmin,Gmax+1):
            for k in range(Gmin,Gmax+1):
                for l in range(Gmin,Gmax+1):
                    if [i,j] not in Gblocked and [k,l] not in Gblocked:
                        states[idx,:] = [i,j,k,l]
                        idx += 1
    return states

def actionSet():
    actions = [[i,j] for i in range(-1,2) for j in range(-1,2)]
    return actions
    
def isValidActionPerPlayer(u,x,Map):
    Gmin, Gmax, Gblocked = Map
    x_new = next_state(x,u)
    # check if the new state is valid
    if (x_new >= Gmin).all() and (x_new <= Gmax).all() and not np.all(x_new == Gblocked):
        return True
    else:
        return False
    
def minimax(A):
    nx = A.shape[0]
    ny = A.shape[1]
    c_y = np.zeros((ny+1,1))
    c_y[-1] = 1
    Aub_y = np.hstack((A, -np.ones((nx,1))))
    bub_y = np.zeros((nx,1))
    Aeq_y = np.ones((1,ny+1))
    Aeq_y[0,-1] = 0
    beq_y = np.array([[1]])
    b = [(0,1),]*ny
    b.append((None,None))
    bounds_y = np.array(b)

    val_y = linprog(c_y, A_ub=Aub_y, b_ub=bub_y, A_eq=Aeq_y, b_eq=beq_y, bounds=bounds_y)

    c_x = np.zeros((nx+1,1))
    c_x[-1] = -1
    Aub_x = np.hstack((-A.T, np.ones((ny,1))))
    bub_x = np.zeros((ny,1))
    Aeq_x = np.ones((1,nx+1))
    Aeq_x[0,-1] = 0
    beq_x = np.array([[1]])
    b = [(0,1),]*nx
    b.append((None,None))
    bounds_x = np.array(b)

    val_x = linprog(c_x, A_ub=Aub_x, b_ub=bub_x, A_eq=Aeq_x, b_eq=beq_x, bounds=bounds_x)
    success = val_y.success and val_x.success and val_y.fun == -val_x.fun
    return success, val_y.fun, val_x.x[0:nx], val_y.x[0:ny] 
    
def instantaneous_cost(x):
    return np.linalg.norm(x[0:2]-x[2:4])

def samplePath(x0, T, U_opt1, U_opt2, states, actions, sample=True):
    x = np.zeros((T+1,4))
    x[0] = x0
    u = np.zeros((T,4))
    u1_prob = np.zeros((T,9))
    u2_prob = np.zeros((T,9))
    for t in range(T):
        u1_prob[t] = U_opt1[t,np.where((states == x[t]).all(axis=1))[0][0],:]
        u2_prob[t] = U_opt2[t,np.where((states == x[t]).all(axis=1))[0][0],:]

        # sample from u1,u2 to create u
        if sample:
            u1 = np.random.choice(9,1,p=u1_prob[t])
            u2 = np.random.choice(9,1,p=u2_prob[t])
            u[t] = np.concatenate((actions[u1[0]],actions[u2[0]]))
        else:
            u1 = np.argmax(u1_prob[t])
            u2 = np.argmax(u2_prob[t])
            u[t] = np.concatenate((actions[u1],actions[u2]))
        x[t+1] = next_state(x[t],u[t])
    return x, u, u1_prob, u2_prob

def backwardDynamicProgramming(states, actions, T, Map):
    V = np.zeros((T+1, Nx))
    RewardT = -100
    V[T,:] = finalValue(RewardT, Nx, Map) # If predator catch prey, then reward is -100

    # Backward dynamic programming
    A = np.ones([Nu,Nu])
    for t in range(T-1,-1,-1):
        for i, x in enumerate(states):
            validActionSet1=[]
            validActionSet2=[]
            saveid2Flag = True
            for id1,u1 in enumerate(actions):
                if isValidActionPerPlayer(u1, x[0:2], Map):
                    for id2,u2 in enumerate(actions):
                        if isValidActionPerPlayer(u2, x[2:4], Map):
                            u = np.concatenate((u1,u2))
                            x_next = next_state(x,u)
                            A[id1,id2] = instantaneous_cost(x) + V[t+1,np.where((states == x_next).all(axis=1))[0][0]]
                            if saveid2Flag == True:
                                validActionSet2.append(id2)
                    validActionSet1.append(id1)
                    saveid2Flag = False
            costs = A[np.ix_(validActionSet1,validActionSet2)]
            success, val, g1, g2 = minimax(costs)
            V[t,i] = val
            U_opt1[t,i,validActionSet1] = g1
            U_opt2[t,i,validActionSet2] = g2

    return V, U_opt1, U_opt2

if __name__ == '__main__':

    # State space
    Gmax = 6
    Gmin = 0
    Gblocked = [[3,3]]
    Map = (Gmin, Gmax, Gblocked)

    # Number of states
    Nx = (Gmax - Gmin + 1)**4 - len(Gblocked)**2

    # Action space per player
    Nu = 3**2

    T = 7 # Time horizon
    # Initialize the policy table
    U_opt1 = np.zeros((T,Nx,Nu))
    U_opt2 = np.zeros((T,Nx,Nu))

    # Table of possible states
    states = stateSet(Nx, Map)
    # Table of possible actions per player
    actions = actionSet()

    # Check if the files exist on disk
    if not os.path.isfile('V.npy') or not os.path.isfile('U_opt1.npy') or not os.path.isfile('U_opt2.npy'):
        solve_load = '1'
    else:
        solve_load =  input("Press 1 to solve the game, 2 to load the solution from disk: \n")

    if solve_load == '1':
        print("\nSolving the game with BDP...")
        # Backward dynamic programming
        V, U_opt1, U_opt2 = backwardDynamicProgramming(states, actions, T, Map)

        # Save matrices to disk
        # Delete the files if they already exist
        try:
            os.remove('V.npy')
            os.remove('U_opt1.npy')
            os.remove('U_opt2.npy')
        except OSError:
            pass

        np.save('V.npy', V)
        np.save('U_opt1.npy', U_opt1)
        np.save('U_opt2.npy', U_opt2)
    else:
        print("\nLoading the solution from disk...")
        # Load matrices from disk
        V = np.load('V.npy')
        U_opt1 = np.load('U_opt1.npy')
        U_opt2 = np.load('U_opt2.npy')

    # A sample optimal path is
    print('\nThe optimal sample path is:')
    x_sample, u_sample, u1_prob, u2_prob = samplePath(states[45,:],T,U_opt1,U_opt2,states,actions,False)
    print("States")
    print(x_sample)
    print("\nActions")
    print(u_sample)
    print("\nProbability of u1")
    print(u1_prob)
    print("\nProbability of u2")
    print(u2_prob)

    # Create the animation
    anim.createAnimation(x_sample, Map)

