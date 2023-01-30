# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cugraph.dask.comms.comms as Comms
from dask.distributed import Client
from dask_cuda import LocalCUDACluster


class LocalCudaClusterManager(object):
    """Manages the state of the LocalCudaCluster"""
    def __init__(self):
        self.cluster = None
        self.client = None
    
    def initialize_local_cluster(self):
        """Initializes the cuda cluster"""
        if self.client is None:
            self.cluster = LocalCUDACluster()
            self.client = Client(self.cluster)
            Comms.initialize(p2p=True)
    
    def get_client(self):
        if self.client is not None:
            return self.client
        else:
            self.initialize_local_cluster()
            return self.client

    def destroy_local_cluster(self) -> None:
        """Destroys the local cuda cluster"""
        if self.client is None:
            return
        Comms.destroy()
        self.client.close()
        self.client = None
        self.cluster.close()
        self.cluster = None
