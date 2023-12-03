# Here we assume a 2D grid world with only 2 players.

# X_t = [X^1_t, X^2_t]; player 1 and player 2 state
# X^i_t = [pos_x; pos_y]; position in the grid world in X and in Y
# U_t = [U^1_t, U^2_t] \in R^4
# Transition: X_{t+1} = X_t + U_t

import numpy as np
from scipy.optimize import linprog
import pdb
import animation as anim 
import concurrent.futures
import argparse
import pickle

terminal_reward = 10 # from the perspective of the prey

def get_terminal_reward(x_pos, terminal_reward):
    # x: (Nx,)
    x1 = x_pos[0 : 2]
    x2 = x_pos[2 : 4]

    if all( x1 != x2 ):
        return terminal_reward
    else:
        return - terminal_reward

def fill_terminal_value_function(V, Nx, N_grid, terminal_reward):
    # Modifies V in-place
    for ii in range(0, Nx):
        x_pos = decode_state(ii, N_grid)
        V[-1, ii] = get_terminal_reward(x_pos, terminal_reward)

def solve_matrix_game(A, x_pos):

    Ny = A.shape[1]
    Nx = A.shape[0]

    # we solve for player 2 first
    c = np.concatenate( ( np.zeros( Ny ), [1.0] ) )
    A_ub = np.c_[ A, -np.ones( Nx ) ]
    b_ub = np.zeros( Nx )

    A_eq = np.concatenate( ( np.ones( Ny ), [0.0] ) )
    A_eq = A_eq[np.newaxis, :] # format required by LP
    b_eq = np.array( [1.0] )

    l_b = np.concatenate( ( np.zeros( Ny ), [ -np.inf ] ) )
    u_b = np.concatenate( ( np.ones( Ny ), [ np.inf ] ) )
    bounds = np.column_stack( ( l_b, u_b ) )

    # solve the linear program for player 2
    res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    # print(res)
    u2_prob = res.x[: -1]

    # now we solve a simpler linear program for player 1
    l = - A @ u2_prob # minus so we min instead of max
    l_b = np.zeros( Nx )
    u_b = np.ones( Nx )
    A_eq = np.ones( ( 1, Nx ) )
    b_eq = np.array( [ 1.0 ] )
    bounds = np.column_stack( ( l_b, u_b) )
    res_2 = linprog(c=l, bounds=bounds, A_eq=A_eq, b_eq=b_eq)
    u1_prob = res_2.x

    value = u1_prob @ ( A @ u2_prob )

    # pdb.set_trace()

    return u1_prob, u2_prob, value

def is_inside_grid(x_pos, N_grid):
    # assumes square 2D grid
    cond_1 = ( x_pos[0] >= 0 ) and ( x_pos[1] >= 0 )
    cond_2 = ( x_pos[0] < N_grid ) and ( x_pos[1] < N_grid)
    if cond_1 and cond_2:
        return True
    else:
        return False

def single_valid(x_pos, u_pos, N_grid):
    x_new = x_pos + u_pos
    return is_inside_grid(x_new, N_grid)

def is_valid_action(x_pos, u1_pos, u2_pos, N_grid):
    x1_pos = x_pos[ 0 : 2 ]
    x2_pos = x_pos[ 2 : 4 ]

    c1 = single_valid(x1_pos, u1_pos, N_grid)
    c2 = single_valid(x2_pos, u2_pos, N_grid)

    return c1 and c2

def get_next_state( x_pos, u1_pos, u2_pos):
    u_pos = np.concatenate( ( u1_pos, u2_pos ) )
    return x_pos + u_pos

def get_instant_reward( x_pos, u1_pos, u2_pos ):
    x1 = x_pos[ 0 : 2 ]
    x2 = x_pos[ 2 : 4 ]
    return np.linalg.norm( x1 - x2, ord=2 )

def encode_state( x_pos, N_grid ):
    row_idx_p1 = x_pos[0]
    col_idx_p1 = x_pos[1]
    x_p1 = row_idx_p1 * N_grid + col_idx_p1

    row_idx_p2 = x_pos[2]
    col_idx_p2 = x_pos[3]
    x_p2 = row_idx_p2 * N_grid + col_idx_p2

    N_meta_grid = N_grid ** 2
    x = x_p1 * N_meta_grid + x_p2

    return x

def decode_state( x, N_grid ):
    # Decode the state to get player indices
    player1_index = x // (N_grid**2)
    player2_index = x % (N_grid**2)

    # Convert player indices to positions
    player1_position = np.array([player1_index // N_grid, player1_index % N_grid])
    player2_position = np.array([player2_index // N_grid, player2_index % N_grid])

    # Combine player positions into a single array
    x_pos = np.concatenate([player1_position, player2_position])

    return x_pos


def decode_action( u ):
    U = np.array([
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, -1],
        [0, 0],
        [0, 1],
        [1, -1],
        [1, 0],
        [1, 1],
    ])
    return U[u, :]

def process_item( x, Nu, N_grid, V ):
    A = [ ]
    u1_values = [ ]
    u2_values = [ ]
    x_pos = decode_state(x, N_grid)
    # print("x: " + str(x))
    for u1 in range(Nu): # for player 1
        u1_pos = decode_action( u1 )
        if single_valid(x_pos[0:2], u1_pos, N_grid):
            u1_values.append(u1)
            curr_col = [ ]
            for u2 in range(Nu): # for player 2
                u2_pos = decode_action( u2 )
                if single_valid(x_pos[2:4], u2_pos, N_grid):

                    y_pos = get_next_state( x_pos, u1_pos, u2_pos )
                    y = encode_state( y_pos, N_grid )
                    curr_val = get_instant_reward( x_pos, u1_pos, u2_pos ) + V[y]
                    curr_col.append(curr_val)

                    if not (u2 in u2_values):
                        u2_values.append(u2)

            if len( curr_col ) != 0:
                A.append( curr_col )
    
    # solve the matrix game
    A = np.array(A) # Check this, I was sure I required a transpose
    u1_prob, u2_prob, r = solve_matrix_game( A, x_pos )

    # Extend probability vector so that it has 9 valus
    u1_prob_full = np.zeros(9)
    u2_prob_full = np.zeros(9)

    u1_prob_full[u1_values] = u1_prob
    u2_prob_full[u2_values] = u2_prob

    return (r, u1_prob_full, u2_prob_full)

    # VV[x] = r
    # U_opt_p1[x, :] = u1_prob_full
    # U_opt_p2[x, :] = u2_prob_full    

def ones_step_bdp_parallel(Nx, Nu, V, N_grid, n_cores):
    VV = np.zeros( Nx )
    # Uopt = np.zeros((Nx, Nu, Nu), dtype=np.int32)
    U_opt_p1 = np.zeros( ( Nx, Nu ), dtype=np.int32 )
    U_opt_p2 = np.zeros( ( Nx, Nu ), dtype=np.int32 )

    x_list = list(range(Nx))
    Nu_list = [Nu] * len(x_list)
    N_grid_list = [N_grid] * len(x_list)
    V_list = [V] * len(x_list)

    with concurrent.futures.ProcessPoolExecutor(n_cores) as  executor:
        results = list( executor.map( process_item, x_list, Nu_list, N_grid_list, V_list))

    # Update the shared array using the results
    for x, value in enumerate(results):
        r = value[0]
        u1_prob_full = value[1]
        u2_prob_full = value[2]

        VV[x] = r
        U_opt_p1[x, :] = u1_prob_full
        U_opt_p2[x, :] = u2_prob_full

    return VV, U_opt_p1, U_opt_p2

    

def one_step_bdp(Nx, Nu, V, N_grid):
    # V is the previous
    VV = np.zeros( Nx )
    # Uopt = np.zeros((Nx, Nu, Nu), dtype=np.int32)
    U_opt_p1 = np.zeros( ( Nx, Nu ), dtype=np.int32 )
    U_opt_p2 = np.zeros( ( Nx, Nu ), dtype=np.int32 )
    # A = np.zeros((Nu, Nu)) # for the matrix game
    # A_u_idx = [ ]
    for x in range(Nx):
        A = [ ]
        u1_values = [ ]
        u2_values = [ ]
        x_pos = decode_state(x, N_grid)
        # print("x: " + str(x))
        for u1 in range(Nu): # for player 1
            u1_pos = decode_action( u1 )
            if single_valid(x_pos[0:2], u1_pos, N_grid):
                u1_values.append(u1)
                curr_col = [ ]
                for u2 in range(Nu): # for player 2
                    u2_pos = decode_action( u2 )
                    if single_valid(x_pos[2:4], u2_pos, N_grid):
                        y_pos = get_next_state( x_pos, u1_pos, u2_pos )
                        y = encode_state( y_pos, N_grid )
                        curr_val = get_instant_reward( x_pos, u1_pos, u2_pos ) + V[y]
                        curr_col.append(curr_val)

                        if not (u2 in u2_values):
                            u2_values.append(u2)

                if len( curr_col ) != 0:
                    A.append( curr_col )
        
        # solve the matrix game
        A = np.array(A) # Check this, I was sure I required a transpose
        u1_prob, u2_prob, r = solve_matrix_game( A, x_pos )

        # Extend probability vector so that it has 9 valus
        u1_prob_full = np.zeros(9)
        u2_prob_full = np.zeros(9)

        u1_prob_full[u1_values] = u1_prob
        u2_prob_full[u2_values] = u2_prob

        VV[x] = r
        U_opt_p1[x, :] = u1_prob_full
        U_opt_p2[x, :] = u2_prob_full

        if all(x_pos == np.array([0, 0, 2, 2])):
            print(u1_prob_full)
            print(u2_prob_full)
    
    return VV, U_opt_p1, U_opt_p2


def run_game(U_t_p1, U_t_p2, x0_pos, N_grid, T):
    x_path = np.zeros((T + 1, 4))
    x_path[0, :] = x0_pos

    for ii in range(0, T):
        print("Stage: ", ii)
        x = int( encode_state(x_path[ii, :], N_grid) )
        print(x_path[ii, :])
        # print('x_ini: ', x_path[ii, :])
        # print(x)
        u1_prob = U_t_p1[ii, x, :]
        u2_prob = U_t_p2[ii, x, :]
        u1 = np.argmax(u1_prob)
        u2 = np.argmax(u2_prob)
        u1_pos = decode_action(u1)
        u2_pos = decode_action(u2)
        u_pos = np.concatenate( (u1_pos, u2_pos ))
        # print('u_pos', u_pos)
        # print('x_pat', x_path[ii, :])
        x_path[ii+1, :] = x_path[ii, :] + u_pos

        # if ii == 4:
        #     pdb.set_trace()

    return x_path

# Main Function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int)
    parser.add_argument("--N_grid", type=int)
    parser.add_argument("--N_cores", type=int)
    parser.add_argument("--terminal_reward", type=float)
    args = vars(parser.parse_args())

    N_grid = args["N_grid"]
    T = args["T"]
    N_cores = args["N_cores"]
    terminal_reward = args['terminal_reward']

    # T = 10
    # Assuming a square grid without obstacles
    # N_grid = 10 # number of blocks in any axis of the grid
    Nx = ( N_grid ** 2) * ( N_grid ** 2)
    Nu = 3* 3 # actions of one of the players
    # N_CORES = 15

    # Value function
    V = -np.ones( ( T + 1, Nx )) # space for value functions
    fill_terminal_value_function(V, Nx, N_grid, terminal_reward)

    U_t_p1 = -np.ones( ( T, Nx, Nu ), dtype=np.int32 ) # space for optimal actions
    U_t_p2 = -np.ones( ( T, Nx, Nu ), dtype=np.int32 ) # space for optimal actions


    for t in range( T, 0, -1 ):
        print("Time: " + str( t ) )
        vv, u_opt_p1, u_opt_p2 = one_step_bdp(Nx, Nu, V[t, :], N_grid)
        # vv, u_opt_p1, u_opt_p2 = ones_step_bdp_parallel(Nx, Nu, V[t, :], N_grid, N_cores)
        V[t-1, :] = vv
        U_t_p1[t-1, :, :] = u_opt_p1
        U_t_p2[t-1, :, :] = u_opt_p2
        # V[T-1, :], Uopt[t-1, :] = one_step_bdp(Nx, Nu, V[t, :])



    # x0_pos = np.array([2, 3, N_grid-1, N_grid-1])
    # positions = run_game(U_t_p1, U_t_p2, x0_pos, N_grid, T)
    # Map = (0, N_grid-1, [])
    # anim.createAnimation(positions, Map)

    # store
    fn = f"T={T}--N_grid={N_grid}.pkl"
    data_store = {
        "U_t_p1": U_t_p1,
        "U_t_p2": U_t_p2,
        "N_grid": N_grid,
        "T": T,
        "terminal_reward": terminal_reward,
    }
    with open(fn, 'wb') as f:
        pickle.dump(data_store, f)
    
    print("Store in: " + fn)