import sys
import subprocess

import torch

def main():
    argslist = list(sys.argv)[1:]
    world_size = torch.cuda.device_count()

    if '--world-size' in argslist:
        argslist[argslist.index('--world-size') + 1] = str(world_size)
    else:
        argslist.append('--world-size')
        argslist.append(str(world_size))

    workers = []

    for i in range(world_size):
        if '--rank' in argslist:
            argslist[argslist.index('--rank') + 1] = str(i)
        else:
            argslist.append('--rank')
            argslist.append(str(i))
        stdout = None if i == 0 else subprocess.DEVNULL
        worker = subprocess.Popen([str(sys.executable)] + argslist, stdout=stdout)
        workers.append(worker)

    returncode = 0
    try:
        for worker in workers:
            worker_returncode = worker.wait()
            if worker_returncode != 0:
                returncode = 1
    except KeyboardInterrupt:
        print('Pressed CTRL-C, TERMINATING')
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.wait()
        raise

    sys.exit(returncode)


if __name__ == "__main__":
    main()
