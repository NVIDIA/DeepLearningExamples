/# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Ensure the docker container has been started the correct way following the instructions in the README, in addition the mount to the outputs directory needs to be passed in."
PATH_TSPP=${1:-"$WORKDIR"} #Path to tspp directory outside of Docker so /home/usr/time-series-benchmark/ instead of /workspace/
python launch_training.py model=tft dataset=electricity trainer/criterion=quantile trainer.config.num_epochs=1 +trainer.config.force_rerun=True hydra.run.dir=/workspace/outputs/0000-00-00/00-00-00/
python launch_inference.py checkpoint=/workspace/outputs/0000-00-00/00-00-00/
cd ${PATH_TSPP}
python launch_triton_configure.py deployment/convert=trt checkpoint=${PATH_TSPP}/outputs/0000-00-00/00-00-00/
python launch_inference_server.py checkpoint=${PATH_TSPP}/outputs/0000-00-00/00-00-00/
python launch_inference.py inference=triton checkpoint=${PATH_TSPP}/outputs/0000-00-00/00-00-00/
docker stop trt_server_cont
