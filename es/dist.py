import numpy as np
import pickle
import redis
import time
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
RESULTS_KEY = "results"
HOST = 'localhost'
PORT = 6379
DB = 0
PASSWORD = ""




def serialize(x):
    """Return a pickled object."""
    return pickle.dumps(x)


def deserialize(x):
    """Return a depickled object."""
    return pickle.loads(x)


class Master(object):
    
    def __init__(self, nworkers):
        self.r = redis.Redis(host=HOST, port=PORT, db=DB, password=PASSWORD)
        self.run_id = 1
        self.nworkers = nworkers

        for key in self.r.scan_iter():
            logger.info(f"deleting key: {key}")
            self.r.delete(key)

    def wait_for_results(self):
        """Wait for all workers to send fitnesses and seed to redis."""
        rewards = []
        seeds = []
        returned = 0
        noiseless = 0
        while returned < self.nworkers:
            _, res = self.r.blpop(RESULTS_KEY)
            rew, seed = deserialize(res)
            """if(seed == -1):
                noiseless = rew
            else:
                rewards.append(rew)
                seeds.append(seed)
                """
            rewards.append(rew)
            seeds.append(seed)
            returned += 1
            time.sleep(0.01)
        return rewards, seeds, noiseless

    def push_run(self, seeds, rewards):
        self.r.set("seeds", serialize(seeds))
        self.r.set("rewards", serialize(rewards))
        self.r.set("run_id", serialize(self.run_id))
        self.run_id += 1
  

class Worker(object):
    def __init__(self, worker_id, lr):
        self.r = redis.Redis(host=HOST, port=PORT, db=DB, password=PASSWORD)
        self.run_id = 0
        self.worker_id = worker_id
        self.learning_rate = lr
    
    def poll_run(self):
            while True:
                new_run_id = deserialize(self.r.get("run_id"))
                time.sleep(0.1)
                if new_run_id != self.run_id:
                    break
            
            self.run_id = new_run_id
            rews = deserialize(self.r.get("rewards"))
            seeds = deserialize(self.r.get("seeds"))
            return rews, seeds
        

    def send_result(self, rew, seed):
        """Put fitnesses and seed in redis."""
        self.r.rpush(RESULTS_KEY, serialize((rew, seed)))