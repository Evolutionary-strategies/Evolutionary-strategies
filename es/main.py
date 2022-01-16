from es.es import run_master, run_worker
import multiprocessing as mp

def launch(nworkers, ismaster):
    if ismaster:
        master = mp.Process(taget = run_master, args = nworkers)
        nworkers -= 1
        
    workers = [mp.Process(target=run_worker) for x in range(0, (nworkers)//5)]
    
