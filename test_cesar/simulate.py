import argparse
import numpy as np
import animation as anim
import pickle
import numpy as np
from solve import run_game

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int)
    parser.add_argument("--N_grid", type=int)
    parser.add_argument("--terminal_reward", type=float)
    parser.add_argument("--x0", type=int, nargs=4)
    args = vars(parser.parse_args())

    N_grid = args["N_grid"]
    T = args["T"]
    terminal_reward = args['terminal_reward']
    x0_pos = np.array(args['x0'])

    # store
    # fn = f"T={T}--N_grid={N_grid}.pkl"
    fn = f"T={T}--N_grid={N_grid}--terminal_reward={int(terminal_reward)}.pkl"
    print("Loading: ", fn)
    with open(fn, 'rb') as f:
        data_dict = pickle.load(f)

    U_t_p1 = data_dict["U_t_p1"]
    U_t_p2 = data_dict["U_t_p2"]
    N_grid = data_dict["N_grid"]
    T = data_dict["T"]
    G_blocked = data_dict["G_blocked"]

    # load a strategy and plot
    positions = run_game(U_t_p1, U_t_p2, x0_pos, N_grid, T)
    if len(G_blocked) == 0:
        Map = (0, N_grid-1, [])
    else:
        Map = (0, N_grid-1, G_blocked.tolist())
    anim.createAnimation(positions, Map)

    print(positions)

