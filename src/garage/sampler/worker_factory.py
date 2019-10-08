"""Worker factory used by Samplers to construct Workers."""
import psutil

from garage.sampler.worker import DefaultWorker


def identity_function(value):
    """Do nothing.

    This function exists so it can be pickled.
    """
    return value


class WorkerFactory:
    """Constructs workers for Samplers.

    The intent is that this object should be sufficient to avoid subclassing
    the sampler. Instead of subclassing the sampler for e.g. a specific
    backend, implement a specialized WorkerFactory (or specify appropriate
    functions to this one). Not that this object must be picklable, since it
    may be passed to workers. However, its fields individually need not be.

    All arguments to this type must be passed by keyword.

    Args:
        seed(int): The seed to use to intialize random number generators.
        n_workers(int): The number of workers to use.
        max_path_length(int): The maximum length paths which will be sampled.
        worker_class(type): Class of the workers. Instances should implement
            the Worker interface.

    """

    def __init__(
            self,
            *,  # Require passing by keyword.
            seed,
            max_path_length,
            n_workers=psutil.cpu_count(logical=False),
            worker_class=DefaultWorker):
        self.seed = seed
        self.n_workers = n_workers
        self.max_path_length = max_path_length
        self.worker_class = worker_class

    def get_worker_broadcast(self, objs, preprocess=identity_function):
        """Take an argument and canonicalize it into a list for all workers.

        This helper function is used to handle arguments in the sampler API
        which may (optionally) be lists. Specifically, these are agent, env,
        agent_update, and env_update. Checks that the number of parameters is
        correct.

        Args:
            objs(object or list): Must be either a single object or a list
                of length self.n_workers.
            preprocess(function): Function to call on each single object before
                creating the list.

        Returns:
            list[object]: A list of length self.n_workers.

        """
        if isinstance(objs, list):
            print('objs', objs)
            if len(objs) != self.n_workers:
                raise ValueError(
                    "Length of list doesn't match number of workers")
            return [preprocess(obj) for obj in objs]
        else:
            obj = preprocess(objs)
            return [obj for _ in range(self.n_workers)]

    def __call__(self, worker_number):
        """Construct a worker given its number."""
        if worker_number >= self.n_workers:
            raise ValueError('Worker number is too big')
        return self.worker_class(self.seed, self.max_path_length,
                                 worker_number)
