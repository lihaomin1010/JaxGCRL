#!/bin/bash

export LD_LIBRARY_PATH=""

# reacher cheetah humanoid
# ant ant_random_start ant_ball ant_push
# ant_ball_maze ant_u_maze  ant_big_maze ant_hardest_maze
# humanoid_u_maze humanoid_big_maze humanoid_hardest_maze
# simple_u_maze simple_big_maze simple_hardest_maze
# pusher_easy pusher_hard pusher_reacher pusher2
# arm_reach arm_grasp arm_push_easy arm_push_hard arm_binpick_easy arm_binpick_hard

# done
# reacher
for seed in 1 ; do
    for env in cheetah humanoid; do
        JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl jaxgcrl crl --env ${env} \
        --log_wandb --wandb_project_name normal --exp-name ${env}-${seed} --wandb_group ${env} \
        --seed ${seed}
    done
    echo "env epoch have finished 1."
done

for seed in 1 ; do
    for env in ant ant_random_start ant_ball ant_push; do
        JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl jaxgcrl crl --env ${env} \
        --log_wandb --wandb_project_name normal --exp-name ${env}-${seed} --wandb_group ${env} \
        --seed ${seed}
    done
    echo "env epoch have finished 2."
done

for seed in 1 ; do
    for env in ant_ball_maze ant_u_maze  ant_big_maze ant_hardest_maze; do
        JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl jaxgcrl crl --env ${env} \
        --log_wandb --wandb_project_name normal --exp-name ${env}-${seed} --wandb_group ${env} \
        --seed ${seed}
    done
    echo "env epoch have finished 3 ."
done

for seed in 1 ; do
    for env in humanoid_u_maze humanoid_big_maze humanoid_hardest_maze; do
        JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl jaxgcrl crl --env ${env} \
        --log_wandb --wandb_project_name normal --exp-name ${env}-${seed} --wandb_group ${env} \
        --seed ${seed}
    done
    echo "env epoch have finished 4."
done

for seed in 1 ; do
    for env in pusher_easy pusher_hard pusher_reacher pusher2; do
        JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl jaxgcrl crl --env ${env} \
        --log_wandb --wandb_project_name normal --exp-name ${env}-${seed} --wandb_group ${env} \
        --seed ${seed}
    done
    echo "env epoch have finished 5."
done

for seed in 1 ; do
    for env in arm_reach arm_grasp arm_push_easy arm_push_hard arm_binpick_easy arm_binpick_hard; do
        JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl jaxgcrl crl --env ${env} \
        --log_wandb --wandb_project_name normal --exp-name ${env}-${seed} --wandb_group ${env} \
        --seed ${seed}
    done
    echo "env epoch have finished 6 ."
done

for seed in 1 ; do
    for env in reacher cheetah humanoid; do
        JAX_TRACEBACK_FILTERING=off XLA_PYTHON_CLIENT_MEM_FRACTION=.95 MUJOCO_GL=egl jaxgcrl crl --env ${env} \
        --log_wandb --wandb_project_name normal --exp-name ${env}-${seed} --wandb_group ${env} \
        --seed ${seed}
    done
    echo "env epoch have finished 7 ."
done
echo "All runs have finished."
