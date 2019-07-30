#!/bin/bash
#SBATCH -p mlperf		# partition
#SBATCH -N 1       		# number of nodes
#SBATCH -t 12:00:00		# wall time
#SBATCH -J image_classification	# job name
#SBATCH --exclusive   		# exclusive node access
#SBATCH --mem=0   		# all mem avail
#SBATCH --mail-type=FAIL        # only send email on failure
#SBATCH --ntasks-per-node=8	# n tasks per machine (one task per gpu)
#SBATCH --threads-per-core=2	# HT is on
#SBATCH --cores-per-socket=20	# 20 cores on each socket 
#SBATCH --overcommit

hostname
#DGXIBDEVICES=$(eval ls /dev/infiniband/ | tr " " "\n" | awk '{printf "--device=/dev/infiniband/%s ",$1}' | sed s'/.$//')
printf "DGXIBDEVICES=%s\n" "$DGXIBDEVICES"
printf "VOLS=%s\n" "$VOLS"
printf "EXTRA_PARAMS=%s\n" "$EXTRA_PARAMS"

cd $CODEDIR

VOLS+=" -v $CHKPTDIR/$SLURM_JOB_ID:/checkpoints"

mkdir -p $CHKPTDIR/$SLURM_JOB_ID

## DO NOT CHANGE ANYTHING BELOW -- DL params are in run_and_time.sh and config_<system>.sh files 

DEBUG=1  # 1 = Print verbose messages for debugging

## Pre-warming the containers ##
hosts=( `scontrol show hostname |tr "\n" " "` )
pids=(); for hostn in ${hosts[@]}; do
  timeout -k 600s 600s \
  srun -N 1 -n 1 -w $hostn \
    docker pull $CONT &
  pids+=($!);
   pids+=($!); rets+=($?);
done
wait "${pids[@]}"
success=0; for s in ${rets[@]}; do ((success+=s)); done ; if [ $success -ne 0 ]; then echo "ERR: Container pull failed"; exit $success ; fi

IBDEVICES=${IBDEVICES:-$DGXIBDEVICES}

## Check whether we are running in a slurm env
INSLURM=1
if [[ -z "$SLURM_JOB_ID" ]]; then
  INSLURM=0
  export SLURM_JOB_ID="${DATESTAMP}"
  export SLURM_NNODES=1
fi
if [[ -z "SLURM_JOB_ID" || $SLURM_NNODES -eq 1 ]]; then
  # don't need IB if not multi-node
  export IBDEVICES=""
fi

# Create results directory
LOGFILE_BASE="${LOGDIR}/${DATESTAMP}"
mkdir -p $(dirname "${LOGFILE_BASE}")

export CONTNAME="${SLURM_JOB_ID}"
export DOCKEREXEC="nvidia-docker run --rm --net=host --uts=host --ipc=host --ulimit stack=67108864 --ulimit memlock=-1 --security-opt seccomp=unconfined  $IBDEVICES"
CMD="python -np $((SLURM_NNODES*DGXNGPU)) -x EXTRA_PARAMS=\"${EXTRA_PARAMS}\" -x NCCL_LL_THRESHOLD=0 -x NCCL_DEBUG=INFO -x NCCL_NET_GDR_READ=1 -x NCCL_SOCKET_IFNAME=^docker0,bond0,lo $BIND ./run_pretraining.sh"
echo $CMD

mkdir -m 777 -p $LOGDIR
echo $CMD | tee -a $LOGDIR/$DATESTAMP.log 
echo "slurm job id" $SLURM_JOB_ID &> $LOGDIR/$DATESTAMP.log 

MASTER_IP=`getent hosts \`hostname\` | cut -d ' ' -f1`
SSH=''
SRUN=''
if [[ $INSLURM -eq 0 ]]; then
  export hosts=( `hostname` )
else
  export hosts=( `scontrol show hostname |tr "\n" " "` )
  SSH='ssh -q -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no $hostn'
  SRUN='srun -N 1 -n 1 -w $hostn'
fi
unique_hosts=( $(echo "${hosts[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' ' ) )
export MASTER_HOST=${hosts[0]}

VARS="-e OMPI_MCA_mca_base_param_files=/dev/shm/mpi/${SLURM_JOB_ID}/mca_params.conf -e EXTRA_PARAMS -e GPUS -e BATCHSIZE -e CONT -e DGXSYSTEM=$DGXSYSTEM -e MASTER_HOST -e MASTER_IP -e SLURM_JOB_NUM_NODES -e SLURM_NNODES -e SLURM_NTASKS_PER_NODE -w /workspace/bert"

RUNSLEEPCMD=""

[[ "${PULL}" -eq "1" ]] && docker pull $CONT

## Setting up MPI
# MPI support files - in /dev/shm/mpi/<jobid>
# 1. Copy user keys to /dev/shm/mpi/<jobid>
# 2. Create mca_params.conf
# 3. Create sshentry.sh to support lauching into containers on worker nodes
# 4. Create mpi_hosts file
# 5. Copy standard ssh

if [[ $SLURM_NNODES -ne "1" ]]; then

  # Make keys and copy
  echo

  [[ $DEBUG == 1 ]] && echo "Setting up ssh keys and config"

  mkdir -p ${HOME}/.ssh/sbatch/${SLURM_JOB_ID}
  ssh-keygen -t rsa -b 2048 -n "" -f "${HOME}/.ssh/sbatch/${SLURM_JOB_ID}/sshkey.rsa" -C "mxnet_${SLURM_JOB_ID}_"  &>/dev/null
  echo command=no-port-forwarding,no-agent-forwarding,no-X11-forwarding $(cat ${HOME}/.ssh/sbatch/${SLURM_JOB_ID}/sshkey.rsa.pub) >> ${HOME}/.ssh/authorized_keys
  chmod 600 ~/.ssh/authorized_keys

  [[ $DEBUG == 1 ]] && echo "Copy keys: srun -n $SLURM_JOB_NUM_NODES  && cp -R ${HOME}/.ssh/sbatch/${SLURM_JOB_ID} /dev/shm/mpi && chmod 700 /dev/shm/mpi/${SLURM_JOB_ID}" 

  srun  -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "mkdir -p /dev/shm/mpi/${SLURM_JOB_ID}; cp -R ${HOME}/.ssh/sbatch/${SLURM_JOB_ID} /dev/shm/mpi; chmod 700 /dev/shm/mpi/${SLURM_JOB_ID}"

  sleep 2 # Making copy

  [[ $DEBUG == 1 ]] && ls /dev/shm

  # Create mpi config file
  srun  -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 tee /dev/shm/mpi/${SLURM_JOB_ID}/mca_params.conf <<EOF
plm_rsh_agent = /usr/bin/ssh
plm_rsh_args = -i /dev/shm/mpi/${SLURM_JOB_ID}/sshkey.rsa -oStrictHostKeyChecking=no -oUserKnownHostsFile=/dev/null -oLogLevel=ERROR -l ${USER}
orte_default_hostfile = /dev/shm/mpi/${SLURM_JOB_ID}/mpi_hosts
btl_openib_warn_default_gid_prefix = 0
mpi_warn_on_fork = 0
allow_run_as_root = 1
EOF

  [[ $DEBUG == 1 ]] && echo "::mca_params.conf=" && cat /dev/shm/mpi/${SLURM_JOB_ID}/mca_params.conf

  # Create ssh helper script that transfers an ssh into a compute node into the running container on that node
  srun -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 tee /dev/shm/mpi/${SLURM_JOB_ID}/sshentry.sh <<EOF
#!/bin/bash
echo "::sshentry: entered \$(hostname)"
[[ -f $CONTNAME ]] && "::worker container not found error" && exit 1
echo "::sshentry: running \$SSH_ORIGINAL_COMMAND"
exec docker exec $CONTNAME /bin/bash -c "\$SSH_ORIGINAL_COMMAND"
EOF

  [[ $DEBUG == 1 ]] && echo "::sshentry=" && cat /dev/shm/mpi/${SLURM_JOB_ID}/sshentry.sh

  # Create mpi hostlist
  for h in ${hosts[@]}; do
     echo "$h slots=${SLURM_NTASKS_PER_NODE}" >> /dev/shm/mpi/${SLURM_JOB_ID}/mpi_hosts
  done

  [[ $DEBUG == 1 ]] && echo '::mpi-host file=' && cat /dev/shm/mpi/${SLURM_JOB_ID}/mpi_hosts

  srun -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 bash -c "cp $(which ssh) /dev/shm/mpi/${SLURM_JOB_ID}/.;  chmod 755 /dev/shm/mpi/${SLURM_JOB_ID}/mca_params.conf;  chmod 755 /dev/shm/mpi/${SLURM_JOB_ID}/sshentry.sh"

  # Check that ssh/mpi dir has correct number of files
  [[ $(ls /dev/shm/mpi/${SLURM_JOB_ID} | wc -w) -lt 5 ]]  && echo "ERR: /dev/shm/mpi/${SLURM_JOB_ID} doesn't exist or missing ssh/mpi files" && exit $?

fi

# Container launch
if [[ $INSLURM -eq 1 ]]; then

  # Launch containers behind srun

  [[ $DEBUG == 1 ]] && echo "" && echo ":Launch containers:  srun  -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 $DOCKEREXEC --name $CONTNAME $VOLS $VARS $CONT bash -c 'sleep infinity'"
  srun  -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 $DOCKEREXEC --name $CONTNAME $VOLS $VARS $CONT bash -c 'sleep infinity' & rv=$?
else
  $DOCKEREXEC --name $CONTNAME $VOLS $VARS $CONT bash -c 'sleep infinity' & rv=$?
fi
[[ $rv -ne 0 ]] && echo "ERR: Launch sleep containers failed." && exit $rv
echo "sleep 60 while we pull our container, good golly!"
sleep 60

# Run benchmarks
echo "sleep again for 20"
sleep 20
export EXTRA_PARAMS

(
# Launching app
echo 
echo "Launching user script on master node:"
  hostn=$MASTER_HOST
  $(eval echo $SSH) docker exec $VARS $CONTNAME $MPICMD ; rv=$?
  [[ $rv -ne 0 ]] && echo "ERR: User script failed." && exit $rv
) |& tee ${LOGFILE_BASE}_$nrun.log

# Clean up (note: on SLURM we skip this, as the epilogue will take care of it)
if [[ $INSLURM -eq 0 ]]; then
  docker rm -f $CONTNAME
fi