
# Copyright 2017 NVIDIA Corporation. All Rights Reserved.
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
# ==============================================================================
import os, subprocess, pexpect, psutil
from getpass import getpass

free_mem = lambda: psutil.virtual_memory().free / pow(1024, 3)

print("Need to purge caches to safely run training set protobuf generation.")
print("Free memory in GB before purging: %s" % free_mem())

if free_mem() < 30.0:

    cmd =  'sudo bash -c "echo 3 > /proc/sys/vm/drop_caches"'
    user = pexpect.spawn('whoami').read().strip()
    child = pexpect.spawn(cmd)

    if user != 'root':
        print("The memory is too full to process the training set.")
        print("Please enter sudo password to purge unused virtual memory.")
        print("If you choose to ignore this, purging will not happen and the app can fail at runtime.")
        purge = raw_input("Would you like to purge caches? Y/n: ")
        if purge.lower() == "y":
            sudo_pass = getpass("Enter sudo password: ")
            prompt = r'\[sudo\] password for %s: ' % user           
            idx = child.expect([prompt, pexpect.EOF], 3) 
            child.sendline(sudo_pass)
        else:
            print("Y/y not selected, will continue without purging, app may fail!")
            child.kill(9)
    child.expect(pexpect.EOF)

print("Free memory now, in GB: %s" % free_mem())
