"""Contains PathBuffers used for DnC and HDnC in order to keep track of separate buffers for each policy."""
import collections
from collections.abc import Sequence
import numpy as np
from garage.replay_buffer import PathBuffer


class MoPPathBuffer(PathBuffer, Sequence):
    def __init__(self, num_buffers, capacity_in_transitions, env_spec=None):

        super().__init__(capacity_in_transitions, env_spec=env_spec)
        self._num_buffers = num_buffers
        self._all_buffers = [
            PathBuffer(capacity_in_transitions, env_spec=env_spec)
            for _ in range(self._num_buffers)
        ]

    def __getitem__(self, i):
        return self._all_buffers[i]

    def __len__(self):
        return len(self._all_buffers)

    def sample_transitions(self, batch_size, buffer_id):
        """
        Samples one set of transitions from buffer_id and one (with only observations) from all buffers
        """

        buffer_sample = self._all_buffers[buffer_id].sample_transitions(batch_size)

        return buffer_sample

    def sample_path(self):
        raise NotImplementedError

    def sample_timesteps(self, batch_size):
        raise NotImplementedError


class DnCPathBuffer(PathBuffer, Sequence):
    def __init__(
        self, num_buffers, capacity_in_transitions, sampling_type, env_spec=None
    ):

        ### ASDF not sure about this
        super().__init__(capacity_in_transitions, env_spec=env_spec)
        self._num_buffers = num_buffers
        self._all_buffers = [
            PathBuffer(capacity_in_transitions, env_spec=env_spec)
            for _ in range(self._num_buffers)
        ]
        self._sampling_type = sampling_type

    def __getitem__(self, i):
        return self._all_buffers[i]

    def __len__(self):
        return len(self._all_buffers)

    def sample_all_transitions(self, batch_size):
        all_obs, all_acs = [], []
        for (i, buffer) in enumerate(self._all_buffers):
            if buffer.n_transitions_stored > 0:
                if self._sampling_type == "all" or self._sampling_type == "i":
                    sample = buffer.sample_transitions(batch_size // self._num_buffers)
                elif self._sampling_type == "j":
                    sample = buffer.sample_transitions(batch_size)
                elif self._sampling_type == "i+j":
                    sample = buffer.sample_transitions(batch_size // 2)
                all_obs.append(sample["observation"])
                all_acs.append(sample["action"])
        return all_obs, all_acs

    def sample_all_other_transitions(self, batch_size, buffer_id):
        # dict_keys(['observation', 'action', 'reward', 'next_observation', 'terminal']) (32,20)

        all_obs, all_acs, all_rews, all_next_obs, all_terms = [], [], [], [], []
        for (i, buffer) in enumerate(self._all_buffers):
            if buffer.n_transitions_stored > 0 and i != buffer_id:
                sample = buffer.sample_transitions(batch_size // (self._num_buffers -1) )
                all_obs.append(sample["observation"])
                all_acs.append(sample["action"])
                all_rews.append(sample["reward"])
                all_next_obs.append(sample["next_observation"])
                all_terms.append(sample["terminal"])

        return {"observation": np.concatenate(all_obs),
                "action": np.concatenate(all_acs),
                "reward": np.concatenate(all_rews),
                "next_observation": np.concatenate(all_next_obs),
                "terminal": np.concatenate(all_terms)}

    def sample_transitions(self, batch_size, buffer_id):
        """
        Samples one set of transitions from buffer_id and one (with only observations) from all buffers
        """

        buffer_sample = self._all_buffers[buffer_id].sample_transitions(batch_size)

        return buffer_sample
        # return all_obs, buffer_sample

    def sample_path(self):
        raise NotImplementedError

    def sample_timesteps(self, batch_size):
        raise NotImplementedError


class HDnCPathBuffer(DnCPathBuffer, Sequence):
    def __init__(self, num_buffers, capacity_in_transitions, env_spec=None):

        ### ASDF not sure about this
        super().__init__(num_buffers, capacity_in_transitions, env_spec=env_spec)
        self._num_buffers = num_buffers
        self._all_buffers = [
            EpisodePathBuffer(capacity_in_transitions, env_spec=env_spec)
            for _ in range(self._num_buffers)
        ]

    def sample_transitions(self, batch_size, buffer_id):
        """
        Samples one set of transitions from buffer_id and one (with only observations) from all buffers
        Additionally, samples initial state (ASDF: eventually context), LL policy id, and return for training HL policy.
        """
        all_obs = []
        for (i, buffer) in enumerate(self._all_buffers):
            if buffer.n_transitions_stored > 0:
                sample = buffer.sample_transitions(batch_size // self._num_buffers)
                all_obs.extend(sample["observation"])
        all_obs = np.array(all_obs)

        buffer_sample = self._all_buffers[buffer_id].sample_transitions(batch_size)

        return all_obs, buffer_sample

    def sample_contexts(self, batch_size):
        all_samples = {"returns": [], "contexts": [], "policy_ids": []}
        for (i, buffer) in enumerate(self._all_buffers):
            if buffer.n_transitions_stored > 0:
                sample = buffer.sample_contexts(batch_size // self._num_buffers)
                all_samples["returns"].extend(sample["returns"])
                all_samples["contexts"].extend(sample["contexts"])
                all_samples["policy_ids"].extend(sample["policy_ids"])

        return all_samples


class EpisodePathBuffer(PathBuffer):
    """
    PathBuffer that keeps track of returns for each episodes, initial state/context, initial action/policy_id
    basically just so there's an easy way to sample data for training high level DnC policy
    """

    def __init__(self, capacity_in_transitions, env_spec=None):
        super().__init__(capacity_in_transitions, env_spec=env_spec)
        self._returns = collections.deque()
        self._contexts = collections.deque()
        self._policy_ids = collections.deque()

    def add_path(self, path):
        """
        Additionally also compute and keep track of returns.
        """

        ### extra stuff, ASDF: or do discounted returns?, also context set to s_0 for now

        ret = np.sum(path["reward"])
        # print(len(path["reward"]), ret)
        context = path["observation"][0]

        ### ASDF hack need to fix to policy_id from policy action infos
        if len(set(path["policy_id"])) != 1:
            import ipdb

            ipdb.set_trace()
        # print(len(path["policy_id"]), set(path["policy_id"]))
        policy_id = path["policy_id"][0]

        for key, buf_arr in self._buffer.items():
            path_array = path.get(key, None)
            if path_array is None:
                raise ValueError("Key {} missing from path.".format(key))
            if len(path_array.shape) != 2 or path_array.shape[1] != buf_arr.shape[1]:
                raise ValueError("Array {} has wrong shape.".format(key))
        path_len = self._get_path_length(path)
        first_seg, second_seg = self._next_path_segments(path_len)
        # Remove paths which will overlap with this one.
        while self._path_segments and self._segments_overlap(
            first_seg, self._path_segments[0][0]
        ):
            self._path_segments.popleft()
            self._returns.popleft()
            self._contexts.popleft()
            self._policy_ids.popleft()
        while self._path_segments and self._segments_overlap(
            second_seg, self._path_segments[0][0]
        ):
            self._path_segments.popleft()
            self._returns.popleft()
            self._contexts.popleft()
        self._path_segments.append((first_seg, second_seg))
        self._returns.append(ret)
        self._contexts.append(context)
        self._policy_ids.append(policy_id)
        for key, array in path.items():
            if key == "policy_id":
                continue
            buf_arr = self._get_or_allocate_key(key, array)
            # numpy doesn't special case range indexing, so it's very slow.
            # Slice manually instead, which is faster than any other method.
            buf_arr[first_seg.start : first_seg.stop] = array[: len(first_seg)]
            buf_arr[second_seg.start : second_seg.stop] = array[len(first_seg) :]
        if second_seg.stop != 0:
            self._first_idx_of_next_path = second_seg.stop
        else:
            self._first_idx_of_next_path = first_seg.stop
        self._transitions_stored = min(
            self._capacity, self._transitions_stored + path_len
        )

    def sample_contexts(self, batch_size):
        assert len(self._returns) == len(self._contexts)
        idx = np.random.randint(len(self._returns), size=batch_size)
        ### ASDF probably inefficient
        return {
            "returns": np.array(self._returns)[idx],
            "contexts": np.array(self._contexts)[idx],
            "policy_ids": np.array(self._policy_ids)[idx],
        }
