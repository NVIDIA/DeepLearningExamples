NV_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-"all"}

# Start TRITON server in detached state
docker run --gpus $NV_VISIBLE_DEVICES --rm -d \
   --shm-size=1g \
   --ulimit memlock=-1 \
   --ulimit stack=67108864 \
   -p8000:8000 \
   -p8001:8001 \
   -p8002:8002 \
   --name triton_server_cont \
   -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
   -v $PWD/results/triton_models:/models \
   nvcr.io/nvidia/tritonserver:20.06-v1-py3 tritonserver --model-store=/models --strict-model-config=false