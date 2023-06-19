"""
Standard MPC for Autonomous Driving
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial
import argparse

from overtaking.mpc_overtaking import MPC
from overtaking.overtaking_env import OvertakingEnv
from overtaking.animation_overtaking import SimVisual

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_video', type=bool, default=False,
                        help="Save the animation as a video file")
    return parser

def run_mpc(env,goal):
    #
    obs=env.reset(goal)
    t, n = 0, 0
    t0 = time.time()
    arrived = False

    while t < env.sim_T:
        t = env.sim_dt * n
        obs , done, info = env.step()

        #vehicle_pos = np.array(info['quad_s0'][0:3])
        vehicle_pos = np.array(info['vehicle_state'][0:2])
        goal_pos = np.array(obs[0:2])
        relative_pos = vehicle_pos - goal_pos
        print("current pos: ", vehicle_pos, "goal pos: ", goal_pos)
        if np.linalg.norm(relative_pos) < 1:
        #if np.linalg.norm(relative_pos) < 2:
                arrived = True
                break
        #t_now = time.time()
        #print(t_now - t0)
        #t0 = time.time()
        n += 1
        update = False
        if np.linalg.norm(relative_pos) < 1e-1:
                arrived = True
        if t > env.sim_T:
                update = True
        yield [info, t, update]
    print("arrvied: ", arrived)


def main():

    args = arg_parser().parse_args()

    plan_T = 5.0   # Prediction horizon for MPC and local planner
    plan_dt = 0.1 # Sampling time step for MPC and local planner
    
    init_param = []
    #init_param.append(np.array([0.0, 0.0, -.5])) # starting point of the ball
    #init_param.append(np.array([0.0, -3])) # starting velocity of the ball
    init_param.append(np.array([-28.0, -3.0])) # starting point of the quadrotor
    init_param.append(np.array([4.0]))
    goal = np.array([28, -3.0, 0, 7.0, 0.0, 0.0])
    #goal = np.array([15, 10, 3.14/2, 0.0, 0.0, 0.0])

    mpc = MPC(T=plan_T, dt=plan_dt)
    env = OvertakingEnv(mpc, plan_T, plan_dt, init_param)

    run_mpc(env,goal)
    #
    sim_visual = SimVisual(env)

    #run_mpc(env)
    run_frame = partial(run_mpc, env, goal)
    ani = animation.FuncAnimation(sim_visual.fig, sim_visual.update, frames=run_frame,
                                  init_func=sim_visual.init_animate, interval=100, blit=True, repeat=False)

    
    if args.save_video:
        writer = animation.writers["ffmpeg"]
        writer = writer(fps=10, metadata=dict(artist='Yubin Wang'), bitrate=1800)
        ani.save("standardMPC_overtaking.mp4", writer=writer)

    #plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()