if [ ! "$(ls | grep -c docker_scripts)" -eq 1 ]; then
  echo "Run this script from root directory. Usage: bash ./docker_scripts/build_docker.sh"
  exit 1
fi

IMG="${IMAGE:=graph_gen}"
docker build . -t ${IMG}
