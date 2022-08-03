import argparse
import os
import pprint

import numpy as np
import torch
from examples import DQN,QRDQN
from examples import wrap_offworld
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import ShmemVectorEnv, DummyVectorEnv
from tianshou.policy import QRDQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger

import gym
import offworld_gym
from offworld_gym.envs.common.channels import Channels
from offworld_gym.envs.common.enums import AlgorithmMode, LearningType
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) # to surpress the warning when running real env

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='OffWorldMonolithDiscreteReal-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.0)
    parser.add_argument('--eps-train-final', type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=25000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num-quantiles', type=int, default=200)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=200)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=100)
    parser.add_argument('--update-per-step', type=float, default= 0.1)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--frames-stack', type=int, default=1)
    parser.add_argument('--resume-path', type=str, default=None)
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='watch the play of pre-trained policy only'
    )
    parser.add_argument('--save_buffer_name', type=str, default="buffer.hdf5")
    parser.add_argument(
        "--load_buffer_name", type=str, default="buffer.hdf5"
    )
    parser.add_argument("--save-interval", type=int, default=4)
    parser.add_argument('--resume', action="store_true")
    return parser.parse_args()


def train_qrdqn(args=get_args()):
    # args.state_shape = env.observation_space.shape or env.observation_space.n
    # args.action_shape = env.action_space.shape or env.action_space.n
    experiment_name = "QRDQN_Real_Example"
    env_mode = {"train":AlgorithmMode.TRAIN,
                "test":AlgorithmMode.TEST}
    env = gym.make(args.task, channel_type=Channels.RGBD, resume_experiment=False,
                        learning_type=LearningType.END_TO_END, algorithm_mode=env_mode["train"], experiment_name=experiment_name)
    args.state_shape = (2,100,100)
    args.action_shape = 4
    channel = args.state_shape[0]
    # should be (N_FRAMES X C) x H x W
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)

    # make environments
    train_envs = DummyVectorEnv(
        [lambda: wrap_offworld(env) for _ in range(args.training_num)]
    )
    test_envs = train_envs
    # seed
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    # define model
    net = QRDQN(*args.state_shape, args.action_shape, args.num_quantiles, args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = QRDQNPolicy(
        net,
        optim,
        args.gamma,
        args.num_quantiles,
        args.n_step,
        target_update_freq=args.target_update_freq
    ).to(args.device)

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # replay buffer: `save_last_obs` and `stack_num` can be removed together
    # when you have enough RAM
    buffer = VectorReplayBuffer(
        args.buffer_size,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=args.frames_stack * channel
    )

    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # log
    log_path = os.path.join(args.logdir, args.task, experiment_name)
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer, train_interval = 50, update_interval = 50, save_interval=args.save_interval)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return False

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 100k steps
        if env_step <= 4e4:
            eps = args.eps_train - env_step / 4e4 * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.set_eps(eps)
        if env_step % 500 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(
            {
                'model': policy.state_dict(),
                'optim': optim.state_dict(),
            }, os.path.join(log_path, 'checkpoint.pth')
        )
        buffer.save_hdf5(os.path.join(log_path, 'train_buffer.hdf5'))

    # watch agent's performance
    def watch():
        print("Setup test envs ...")
        policy.eval()
        policy.set_eps(args.eps_test)
        test_envs.seed(args.seed)
        if args.save_buffer_name:
            print(f"Generate buffer with size {args.buffer_size}")
            buffer = VectorReplayBuffer(
                args.buffer_size,
                buffer_num=len(test_envs),
                ignore_obs_next=True,
                save_only_last_obs=True,
                stack_num=args.frames_stack
            )
            collector = Collector(policy, test_envs, buffer, exploration_noise=True)
            result = collector.collect(n_step=args.buffer_size)
            print(f"Save buffer into {args.save_buffer_name}")
            # Unfortunately, pickle will cause oom with 1M buffer size
            buffer.save_hdf5(args.save_buffer_name)
        else:
            print("Testing agent ...")
            test_collector.reset()
            result = test_collector.collect(
                n_episode=10, render=args.render
            )
        rew = result["rews"].mean()
        print(f'Mean reward (over {result["n/ep"]} episodes): {rew}')

    if args.watch:
        watch()
        exit(0)

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint['model'])
            policy.optim.load_state_dict(checkpoint['optim'])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")
        buffer_path = os.path.join(log_path, 'train_buffer.hdf5')
        if os.path.exists(buffer_path):
            # train_collector.buffer = pickle.load(open(buffer_path, "rb"))
            buffer = VectorReplayBuffer.load_hdf5(buffer_path)
            print("Successfully restore buffer.")
        else:
            print("Fail to restore buffer.")

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # trainer
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        5 , #args.test_num,
        args.batch_size,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger,
        update_per_step=args.update_per_step,
        resume_from_log = args.resume,
        save_checkpoint_fn=save_checkpoint_fn
    )

    pprint.pprint(result)
    watch()


if __name__ == '__main__':
    train_qrdqn(get_args())
