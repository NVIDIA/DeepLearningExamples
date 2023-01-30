if [ ! "$(ls | grep -c docker_scripts)" -eq 1 ]; then
  echo "Run this script from root directory. Usage: bash ./docker_scripts/run_docker_interactive.sh"
  exit 1
fi

IMG="${IMAGE:=graph_gen}"

nvidia-docker run --rm -it \
  --ipc=host \
  --net=host \
  -v "$(pwd)":/workspace \
  ${IMG} \
  bash
