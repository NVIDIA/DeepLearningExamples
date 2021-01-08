import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from subprocess import call

from data_preprocessing.configs import task

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--task", type=str, required=True, help="Task to download")
parser.add_argument("--results", type=str, default="/data", help="Directory for data storage")

if __name__ == "__main__":
    args = parser.parse_args()
    tar_file = task[args.task] + ".tar"
    file_path = os.path.join(args.results, tar_file)
    call(f"aws s3 cp s3://msd-for-monai-eu/{tar_file} --no-sign-request {args.results}", shell=True)
    call(f"tar -xf {file_path} -C {args.results}", shell=True)
    call(f"rm -rf {file_path}", shell=True)
