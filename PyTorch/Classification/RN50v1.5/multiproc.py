import torch
import sys
import subprocess
import time

argslist = list(sys.argv)[1:]
world_size = torch.cuda.device_count()

if '--world-size' in argslist:
    world_size = min(world_size, int(argslist[argslist.index('--world-size')+1]))
    argslist[argslist.index('--world-size')+1] = str(world_size)
else:
    argslist.append('--world-size')
    argslist.append(str(world_size))

workers = []

for i in range(world_size):
    if '--rank' in argslist:
        argslist[argslist.index('--rank')+1] = str(i)
    else:
        argslist.append('--rank')
        argslist.append(str(i))
    stdout = None if i == 0 else open("GPU_"+str(i)+".log", "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
    workers.append(p)


try:
    up = True
    error = False
    while up and not error:
        up = False
        for p in workers:
            ret = p.poll()
            if ret is None:
                up = True
            elif ret != 0:
                error = True
        time.sleep(0.1)

    if error:
        for p in workers:
            if p.poll() is None:
                p.terminate()
        exit(1)

except KeyboardInterrupt:
    for p in workers:
        p.terminate()
    raise

