import numpy as np

def one_step_bdp(Nx, Nu, V):
    VV = np.zeros(Nx)
    Uopt = np.zeros(Nx,dtype=np.uint32)

    for x in range(1,Nx+1): 
        vn = []
        uv = []
        for u in range(1,Nu+1):
            if isvalid_action(u,x):
                y = next_state(x,u)
                vn.append(instantaneous_cost(x,u)+ V[y-1])
                uv.append(u)
        umin = np.argmin(vn)
        VV[x-1] = vn[umin]
        Uopt[x-1] = uv[umin]

    return VV, Uopt


# Main code starts here
T=5   # horizon
Nx=32  # size of state set
Nu=11  # size of action set

V = -np.ones((T+1,Nx))  # allocate space for value functions 
Uopt = -np.ones((T,Nx),dtype=np.uint32) # allocate space for optimal actions

# initialize appropriately the terminal costs
V[T,:] = ...

for t in range(T, 0, -1):
    V[t-1,:], Uopt[t-1,:] = one_step_bdp(Nx,Nu,V[t,:])

# Trace the state with the optimal policy
x = ...  # put here the initial state
xd = decode_state(x)  # presented in a human-readable form
print("xd= ",xd)
for t in range(T):
    u = Uopt[t, x-1]  # optimal action for state x
    ud = decode_action(u)  # present it in human-readable form
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