import abc
from collections import deque
import h5py
import pickle

import numpy as np
from typing import Dict, Tuple

from vd4rl_utils import ExtendedTimeStep, step_type_lookup


def get_timestep_from_idx(offline_data: dict, idx: int):
    return ExtendedTimeStep(
        step_type=step_type_lookup[offline_data['step_type'][idx]],
        reward=offline_data['reward'][idx],
        observation=offline_data['observation'][idx],
        discount=offline_data['discount'][idx],
        action=offline_data['action'][idx]
    )


class AbstractReplayBuffer(abc.ABC):
    @abc.abstractmethod
    def add(self, time_step):
        pass

    @abc.abstractmethod
    def __next__(self, ):
        pass

    @abc.abstractmethod
    def __len__(self, ):
        pass


class EfficientReplayBuffer(AbstractReplayBuffer):
    '''Fast + efficient replay buffer implementation in numpy.'''

    def __init__(self, buffer_size, batch_size, nstep, discount, frame_stack,
                 data_specs=None, sarsa=False):
        self.buffer_size = buffer_size
        self.data_dict = {}
        self.index = -1
        self.traj_index = 0
        self.frame_stack = frame_stack
        self._recorded_frames = frame_stack + 1
        self.batch_size = batch_size
        self.nstep = nstep
        self.discount = discount
        self.full = False
        self.discount_vec = np.power(discount, np.arange(nstep))  # n_step - first dim should broadcast
        self.next_dis = discount ** nstep
        self.sarsa = sarsa

    def _initial_setup(self, time_step):
        self.index = 0
        self.obs_shape = list(time_step.observation.shape)
        self.ims_channels = self.obs_shape[0] // self.frame_stack
        self.act_shape = time_step.action.shape

        self.obs = np.zeros([self.buffer_size, self.ims_channels, *self.obs_shape[1:]], dtype=np.uint8)
        self.act = np.zeros([self.buffer_size, *self.act_shape], dtype=np.float32)
        self.rew = np.zeros([self.buffer_size], dtype=np.float32)
        self.dis = np.zeros([self.buffer_size], dtype=np.float32)
        self.valid = np.zeros([self.buffer_size], dtype=np.bool_)

    def add_data_point(self, time_step):
        first = time_step.first()
        latest_obs = time_step.observation[-self.ims_channels:]
        if first:
            end_index = self.index + self.frame_stack
            end_invalid = end_index + self.frame_stack + 1
            if end_invalid > self.buffer_size:
                if end_index > self.buffer_size:
                    end_index = end_index % self.buffer_size
                    self.obs[self.index:self.buffer_size] = latest_obs
                    self.obs[0:end_index] = latest_obs
                    self.full = True
                else:
                    self.obs[self.index:end_index] = latest_obs
                end_invalid = end_invalid % self.buffer_size
                self.valid[self.index:self.buffer_size] = False
                self.valid[0:end_invalid] = False
            else:
                self.obs[self.index:end_index] = latest_obs
                self.valid[self.index:end_invalid] = False
            self.index = end_index
            self.traj_index = 1
        else:
            np.copyto(self.obs[self.index], latest_obs)  # Check most recent image
            np.copyto(self.act[self.index], time_step.action)
            self.rew[self.index] = time_step.reward
            self.dis[self.index] = time_step.discount
            self.valid[(self.index + self.frame_stack) % self.buffer_size] = False
            if self.traj_index >= self.nstep:
                self.valid[(self.index - self.nstep + 1) % self.buffer_size] = True
            self.index += 1
            self.traj_index += 1
            if self.index == self.buffer_size:
                self.index = 0
                self.full = True

    def add(self, time_step):
        if self.index == -1:
            self._initial_setup(time_step)
        self.add_data_point(time_step)

    def __next__(self, ):
        indices = np.random.choice(self.valid.nonzero()[0], size=self.batch_size)
        return self.gather_nstep_indices(indices)

    def gather_nstep_indices(self, indices):
        n_samples = indices.shape[0]
        all_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i] + self.nstep)
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        gather_ranges = all_gather_ranges[:, self.frame_stack:]  # bs x nstep
        obs_gather_ranges = all_gather_ranges[:, :self.frame_stack]
        nobs_gather_ranges = all_gather_ranges[:, -self.frame_stack:]

        all_rewards = self.rew[gather_ranges]

        # Could implement below operation as a matmul in pytorch for marginal additional speed improvement
        rew = np.sum(all_rewards * self.discount_vec, axis=1, keepdims=True)

        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        nobs = np.reshape(self.obs[nobs_gather_ranges], [n_samples, *self.obs_shape])

        act = self.act[indices]
        dis = np.expand_dims(self.next_dis * self.dis[nobs_gather_ranges[:, -1]], axis=-1)

        if self.sarsa:
            nact = self.act[indices + self.nstep]
            return (obs, act, rew, dis, nobs, nact)

        return (obs, act, rew, dis, nobs)

    def __len__(self):
        if self.full:
            return self.buffer_size
        else:
            return self.index

    def get_train_and_val_indices(self, validation_percentage):
        all_indices = self.valid.nonzero()[0]
        num_indices = all_indices.shape[0]
        num_val = int(num_indices * validation_percentage)
        np.random.shuffle(all_indices)
        val_indices, train_indices = np.split(all_indices,
                                              [num_val])
        return train_indices, val_indices

    def get_obs_act_batch(self, indices):
        n_samples = indices.shape[0]
        obs_gather_ranges = np.stack([np.arange(indices[i] - self.frame_stack, indices[i])
                                      for i in range(n_samples)], axis=0) % self.buffer_size
        obs = np.reshape(self.obs[obs_gather_ranges], [n_samples, *self.obs_shape])
        act = self.act[indices]
        return obs, act


def load_offline_dataset_into_buffer(offline_dir, replay_buffer, frame_stack, replay_buffer_size):
    filenames = sorted(offline_dir.glob('*_train.pkl'))
    num_steps = 0
    for filename in filenames:
        try:
            with open(filename, 'rb') as f:
                episodes = pickle.load(f)
            # episodes = h5py.File(filename, 'r')
            # episodes = {k: episodes[k][:] for k in episodes.keys()}
            add_offline_data_to_buffer(episodes, replay_buffer, framestack=frame_stack)
            length = episodes['reward'].shape[0]
            num_steps += length
        except Exception as e:
            print(f'Could not load episode {str(filename)}: {e}')
            continue
        print("Loaded {} offline timesteps so far...".format(int(num_steps)))
        if num_steps >= replay_buffer_size:
            break
    print("Finished, loaded {} timesteps.".format(int(num_steps)))


def add_offline_data_to_buffer(offline_data: dict, replay_buffer: EfficientReplayBuffer, framestack: int = 3):
    offline_data_length = offline_data['reward'].shape[0]
    for v in offline_data.values():
        assert v.shape[0] == offline_data_length
    for idx in range(offline_data_length):
        time_step = get_timestep_from_idx(offline_data, idx)
        if not time_step.first():
            stacked_frames.append(time_step.observation)
            time_step_stack = time_step._replace(observation=np.concatenate(stacked_frames, axis=0))
            replay_buffer.add(time_step_stack)
        else:
            stacked_frames = deque(maxlen=framestack)
            while len(stacked_frames) < framestack:
                stacked_frames.append(time_step.observation)
            time_step_stack = time_step._replace(observation=np.concatenate(stacked_frames, axis=0))
            replay_buffer.add(time_step_stack)