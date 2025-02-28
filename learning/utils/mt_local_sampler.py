from garage.sampler import LocalSampler
from garage import EpisodeBatch


class MTLocalSampler(LocalSampler):
    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Collects num_samples / n_workers samples from each worker"""
        self._update_workers(agent_update, env_update)
        batches = []
        num_samples = num_samples / len(self._workers)
        for worker in self._workers:
            completed_samples = 0
            while True:
                batch = worker.rollout()
                completed_samples += len(batch.actions)
                batches.append(batch)
                if completed_samples >= num_samples:
                    break
        samples = EpisodeBatch.concatenate(*batches)
        self.total_env_steps += sum(samples.lengths)
        return samples
