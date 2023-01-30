if [ ! "$(ls | grep -c docker_scripts)" -eq 1 ]; then
  echo "Run this script from root directory. Usage: bash ./docker_scripts/run_docker_notebook.sh"
  exit 1
fi

IMG="${IMAGE:=graph_gen}"

CMD='cd /workspace && echo -e "\nOPEN http://<your_ip>:9916/tree/demos and copy token\n\n"  && jupyter notebook --ip=0.0.0.0 --port=9916'

nvidia-docker run --rm -it \
  --ipc=host \
  --net=host \
  -v "$(pwd)":/workspace \
  ${IMG} \
  bash -c "${CMD}"

# OPEN http://<your_ip>:9916/tree/demos
