# Here we assume a 2D grid world with only 2 players.

# X_t = [X^1_t, X^2_t]; player 1 and player 2 state
# X^i_t = [pos_x; pos_y]; position in the grid world in X and in Y
# U_t = [U^1_t, U^2_t] \in R^4
# Transition: X_{t+1} = X_t + U_t

import numpy as np

TERMINAL_REWARD = 100 # from the perspective of the prey

def get_terminal_reward(x_pos):
    # x: (Nx,)
    x1 = x_pos[0 : 2]
    x2 = x_pos[2 : 4]

    if all( x1 != x2 ):
        return TERMINAL_REWARD
    else:
        return - TERMINAL_REWARD

def encode_state(x_pos):
    # TO BE IMPLEMENTED
    # Input: stacked position of the players on the grid
    # Output: index that correspond to that position, we count from 0
    x = 4
    return x

def decode_state(x):
    # TO BE IMPLEMENTED
    # Input: index that correspond to that position, we count from 0
    # Output: stacked position of the players on the grid
    x_pos = np.array([1, 2, 3, 4])
    return x_pos

def fill_terminal_value_function(V, Nx):
    # Modifies V in place
    for ii in range(0, Nx):
        x_pos = decode_state(ii)
        V[T, ii] = get_terminal_reward(x_pos)
    
def one_step_bdp(Nx, Nu, V):
    # V is the previous
    VV = np.zeros(Nx)
    Uopt = np.zeros(Nx, dtype=np.uint32)

    for x in range(Nx):
        vn = []
        uv = []
        for u in range(Nu):
            u_pos = decode_action(u)
            x_pos = decode_state(x)
            if is_valid_action(u_pos, x_pos):
                y_pos = get_next_state(x_pos, u_pos)
                y = encode_state(y_pos)
                vn.append( get_instant_reward( x_pos, u_pos ) + V[y] )
                uv.append( u )
                



T = 4

# Assuming a square grid without obstacles
N_grid = 4 # number of blocks in any axis of the grid
Nx = ( N_grid ** 2) * ( N_grid ** 2)
Nu = 3* 3 * 2 # at each t only one player acts, check later

# Value function
V = -np.ones( ( T + 1, Nx )) # space for value functions
Uopt = -np.ones( ( T, Nx ), dtype=np.uint32 ) # space for optimal actions
fill_terminal_value_function(V, Nx)

for t in range( T, 0, -1 ):
    V[T-1, :], Uopt[t-1, :] = one_step_bdp(Nx, Nu, V[t, :])