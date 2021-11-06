## Examples

### Available implementations

#### Real

| Algorithm      | Environment            | RL Framework      | Code |
| ---            | ---                    | ---               | ---  |
| Manual control | Monolith Discrete      | -                 | [`examples/real/manual_control.py`](https://github.com/offworld-projects/offworld-gym/blob/main/examples/real/manual_control.py) |
| Random agent   | Monolith Discrete      | -                 | [`examples/real/random_monolith_discrete_real.py`](https://github.com/offworld-projects/offworld-gym/blob/main/examples/real/random_monolith_discrete_real.py) |
| Double DQN     | Monolith Discrete      | Keras RL          | [`examples/real/ddqn_kerasrl_monolith_discrete_real.py`](https://github.com/offworld-projects/offworld-gym/blob/main/examples/real/ddqn_kerasrl_monolith_discrete_real.py) |

#### Sim

| Algorithm      | Environment            | RL Framework      | Code |
| ---            | ---                    | ---               | ---  |
| Random agent   | Monolith Discrete      | -                 | [`examples/sim/random_monolith_discrete_sim.py`](https://github.com/offworld-projects/offworld-gym/blob/main/examples/sim/random_monolith_discrete_sim.py) |
| Random agent   | Monolith Continuous    | -                 | [`examples/sim/random_monolith_continuous_sim.py`](https://github.com/offworld-projects/offworld-gym/blob/main/examples/sim/random_monolith_continuous_sim.py) |
| Double DQN     | Monolith Discrete      | Keras RL          | [`examples/sim/ddqn_kerasrl_monolith_discrete_sim.py`](https://github.com/offworld-projects/offworld-gym/blob/main/examples/real/ddqn_kerasrl_monolith_discrete_sim.py) |


### Additional dependecies
Example implementations in this section are using different RL frameworks, in order a script that relies third-party framework please follow the installation instructions in the corresponding repositories:

  * [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
  * [Tianshou](https://github.com/thu-ml/tianshou)
  * [Keras-RL](https://github.com/offworld-projects/keras-rl/tree/offworld-gym) (note that this is a branch in a fork)
