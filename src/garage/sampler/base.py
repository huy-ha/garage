"""Base sampler class."""

import abc
import copy


class Sampler(abc.ABC):
    """Base class of all samplers."""

    def __init__(self, algo, env):
        """Construct a Sampler from an Algorithm.

        Calling this method is deprecated.
        """
        self.algo = algo
        self.env = env

    @classmethod
    def construct(cls, worker_factory, agents, envs):
        """Construct this sampler.

        Args:
            worker_factory(WorkerFactory): Pickleable factory for creating
                workers. Should be transmitted to other processes / nodes where
                work needs to be done, then workers should be constructed
                there.
            agents(Agent or List[Agent]): Agent(s) to use to perform rollouts.
                If a list is passed in, it must have length exactly
                `worker_factory.n_workers`, and will be spread across the
                workers.
            envs(gym.Env or List[gym.Env]): Environment rollouts are performed
                in. If a list is passed in, it must have length exactly
                `worker_factory.n_workers`, and will be spread across the
                workers.

        Returns:
            Sampler: An instance of `cls`.

        """
        # This implementation works for most current implementations.
        # Relying on this implementation is deprecated, but calling this method
        # is not.
        fake_algo = copy.copy(worker_factory)
        fake_algo.policy = agents
        return cls(fake_algo, envs)

    def start_worker(self):
        """Initialize the sampler.

        i.e. launching parallel workers if necessary.

        This method is deprecated, please launch workers in construct instead.
        """

    @abc.abstractmethod
    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Collect at least a given number transitions (timesteps).

        Args:
            itr(int): The current iteration number. Using this argument is
                deprecated.
            num_samples(int): Minimum number of transitions / timesteps to
                sample.
            agent_update(object): Value which will be passed into the
                `agent_update_fn` before doing rollouts. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update(object): Value which will be passed into the
                `env_update_fn` before doing rollouts. If a list is passed in,
                it must have length exactly `factory.n_workers`, and will be
                spread across the workers.

        Returns:
            List[Dict[str, np.array]]: A list of paths.

        """

    @abc.abstractmethod
    def shutdown_worker(self):
        """Terminate workers if necessary."""
