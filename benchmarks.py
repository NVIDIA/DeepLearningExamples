#!/usr/bin/env python
import yaml
import click
import os
import subprocess
import json
import csv
from datetime import datetime
import pandas as pd
import docker
from gpustat.core import GPUStatCollection
import time
from itertools import product
from mergedeep import merge
import copy
import multiprocessing as mp
import wandb

import sqlite3
from sqlite3 import Error
from sqlalchemy import create_engine

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return conn

def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def update_task_status(conn, benchmark_name, status):
    try:
        c = conn.cursor()
        c.execute("""UPDATE tasks SET status = ? WHERE benchmark_name = ?""", (status, benchmark_name))
    except Error as e:
        print(e)

def clean_wandb_id(run_id, config):
    try:
        wandb.Api().run(f"{config['wandb']['user']}/{config['wandb']['project']}/{run_id}").delete()
        return True
    except Exception as e: 
        print(e)
        return False

def get_config(benchmark_config_file):
    with open(benchmark_config_file, "r") as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return config

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    
    for instance in product(*vals):
        yield dict(zip(keys, instance))



def generate_all_benchmarks(config, interactive_mode, tracking_db):
    templates_ref = config['benchmarks-template']
    for system_name, system_config in config["systems"].items():
        
        if not system_config["active"]:
            print(f"Skipping Benchmarks for {system_name}")
        else:
            print(config["benchmarks"].items())
            for benchmark_name, benchmark_config in config["benchmarks"].items():
                print("")
                print("")
                print(benchmark_name)
                print("")
                print("")
                benchmark_template = templates_ref[benchmark_config['benchmark-template']]
                
                merge(benchmark_template, benchmark_config)
                this_benchmark_config = benchmark_template
                
                if this_benchmark_config['active']:
                    print(f"Running {benchmark_name} for {system_name}")

                    build_docker(this_benchmark_config['docker'], benchmark_name)
                    
                    # if params item is a list => do the cardinality
                    benchmarks_params = list(product_dict(**this_benchmark_config['params']))
                    
                    for benchmark_params in benchmarks_params:
                        current_benchmark_config = copy.copy(this_benchmark_config)
                        
                        current_benchmark_config['params'] = benchmark_params
                        print(f"Generating benchmark for {current_benchmark_config['params']}")
                        generate_docker(system_name, system_config, benchmark_name, current_benchmark_config, config["data"], config["wandb"], interactive_mode, tracking_db)
                else:
                    print(f"Skipping {benchmark_name} for {system_name}")

def build_docker(docker_config, tag):
    cmd = f"cd {docker_config['path']} && docker build -t {tag} -f {docker_config['dockerfile']} ."
    print(f"Building docker for {docker_config['path']} - {tag}")
    os.system(cmd)

def generate_docker(system_name, system_config, benchmark_name, benchmark_config, data_config, wandb_config, interactive_mode, tracking_db):
    NB_CUDA_DEVICES = len(system_config['devices-ids'])
    devices_ids = [str(id) for id in system_config['devices-ids']]
    NVIDIA_VISIBLE_DEVICES=",".join(devices_ids)


    experiments = list()
    mounts = ""
    for data_source, mount_point in benchmark_config['docker']['mounts'].items():
        mounts += f"-v {data_config[data_source]}:{mount_point} "
    
    for capability, is_active in system_config['compute-capabilities'].items():
        if is_active:

            print(f"Benchmarking {capability} for {system_name} and {benchmark_name} batch-size:{benchmark_config['params']['batch-size']} epochs:{benchmark_config['params']['epochs']}")

            extra_replacements = {
                    'NVIDIA_VISIBLE_DEVICES': NVIDIA_VISIBLE_DEVICES,
                    'NB_CUDA_DEVICES':NB_CUDA_DEVICES,
                    'SYSTEM_NAME': system_name,
                    'BENCHMARK_NAME': benchmark_name,
                    'CAPABILITY': capability
                    }

            cmd_replacements = {**benchmark_config['params'], **extra_replacements}
        
            benchmark_cmd = ""
            if 'preparation' in benchmark_config:
                for preparation in benchmark_config['preparation']:
                    benchmark_cmd += preparation.format(**cmd_replacements) + " && "

            if capability not in benchmark_config['docker']['executable']['commands']:
                print(f"Skipping {capability} for {system_name}-{benchmark_name}")
            else:
                benchmark_cmd += benchmark_config['docker']['executable']['commands'][capability].format(**cmd_replacements)

                if interactive_mode:
                    run_mode = f"it -v $PWD/{benchmark_config['docker']['path']}:{benchmark_config['docker']['executable']['path']}"
                else:
                    run_mode = "d"

                run_name = f"{benchmark_name}-{system_name}-{capability}-{cmd_replacements['model']}-B{cmd_replacements['batch-size']}xE{cmd_replacements['epochs']}xLR{cmd_replacements['learning-rate']}"
                
                date = datetime.now().strftime("%Y.%m.%d-%H:%M")
                wandb = ""
                WANDB_NOTES=json.dumps(cmd_replacements)

                memory_info = list()
                for item in ['Size', 'Speed', 'Manufacturer', 'Type', 'Configured Memory Speed', 'Form Factor']:
                    value = subprocess.check_output(f"dmidecode --type 17 | grep '{item}' | head -n 1", shell=True, text=True).strip().replace(f"{item}: ", "")
                    key = f"mem_info_{item.lower().replace(' ','_')}"
                    
                    for unit in ["MB", "MT/s"]:
                        if unit in value:
                            value = value.replace(unit, '').strip()
                            key += f"_{unit.replace('/','_per_')}"

                    memory_info.append(f"{key}={value}")
                
                tags =  ','.join(memory_info)
                if wandb_config["active"]:
                    additional_tags = ','.join(wandb_config['additional-tags'])
                    tags += ',' + additional_tags
                    wandb = f"-e WANDB_API_KEY={wandb_config['key']} -e WANDB_NAME='{run_name}-{date}' -e WANDB_TAGS='{tags}' -e WANDB_NOTES='{WANDB_NOTES}' -e WANDB_ENTITY={wandb_config['user']} -e WANDB_PROJECT={wandb_config['project']}"

                cmd = f"docker run -{run_mode} --rm --ipc=host --name={run_name} {wandb} {mounts} -e run_name={run_name} --gpus 'device={NVIDIA_VISIBLE_DEVICES}' -w {benchmark_config['docker']['executable']['path']} -e PRECISION={capability} {benchmark_name} bash -c '{benchmark_cmd}'"
                print("============")
                print(f"== {run_name} ==")
                experiments.append([
                    run_name,
                    system_name,
                    NVIDIA_VISIBLE_DEVICES,
                    run_name,
                    "PENDING",
                    cmd
                    ])
                print("============")
    
    engine = create_engine(f'sqlite:///{tracking_db}', echo=False)
    df = pd.DataFrame(experiments, columns=headers)
    df.to_sql('tasks', con=engine, if_exists='append')

def get_docker_status(docker_name):
    status = None

def runner(tracking_db, interactive_mode, show_cmd):
    db_conn = create_connection(tracking_db)
    while True:
        run_cycle(db_conn, interactive_mode, show_cmd)
        time.sleep(60)

def run_cycle(db_conn, interactive_mode, show_cmd):
    df = pd.read_sql_query("SELECT * FROM tasks", db_conn)
    client = docker.from_env()
    running_containers_list = list()
    gpus_status = GPUStatCollection.new_query().jsonify()
    gpus_status_dict = dict()

    for gpu in gpus_status['gpus']:
        gpus_status_dict[gpu['index']] = len(gpu['processes'])

    for container in client.containers.list():
        print(f"found {container.name}")
        update_task_status(db_conn, container.name, 'RUNNING')
        running_containers_list.append(container.name)

    for index, row in df.iterrows():
        print(f"{row['docker_name']}1: {row['status']}")
        if row['status'] == 'RUNNING' and row['docker_name'] not in running_containers_list:
            update_task_status(db_conn, container.name, 'STOPPED')

        if row['status'] == 'PENDING':
            gpus_status = GPUStatCollection.new_query().jsonify()
            gpus_status_dict = dict()
            for gpu in gpus_status['gpus']:
                gpus_status_dict[gpu['index']] = len(gpu['processes'])

            gpus = str(row['devices']).split(',')

            nb_processes = 0
            for gpu in gpus:
                nb_processes += gpus_status_dict[int(gpu)]
                print(gpus_status_dict)

            
            if nb_processes == 0:
                print(f"STARTING: {row['docker_name']}")
                cmd = row['cmd']
                if interactive_mode:
                    cmd = cmd.replace(' -d ', ' -it ')
                if show_cmd:
                    print()
                    print(cmd)
                    print()
                os.system(cmd)
                update_task_status(db_conn, row['docker_name'], 'STARTING')
                time.sleep(60)


@click.command()
@click.option('--benchmark_config_file', default="benchmarks.yml", help="YAML config benchmark file")
@click.option('--interactive_mode', is_flag=True, default=False, help="Generate docker run iteractive")
@click.option('--kill_all', is_flag=True, default=False, help="Kill all running containers")
@click.option('--tracking_db', default="tracking.db", help="Benchmark tracking sqlit3 db file")
@click.option('--reset_tracking_db', is_flag=True, default=False, help="Reset Benchmark tracking db")
@click.option('--run', is_flag=True, default=False, help="Run benchmark")
@click.option('--run_only', is_flag=True, default=False, help="Run benchmark without generating tracking file")
@click.option('--skip_generate_tracking', is_flag=True, default=False, help="Generate Tracking benchmark")
@click.option('--clean_wandb', is_flag=True, default=False, help="Clean all wandb run for project")
@click.option('--show_cmd', is_flag=True, default=False, help="Show the run command when a run is started")
def main(benchmark_config_file, interactive_mode, kill_all, tracking_db, reset_tracking_db, run, run_only, 
            skip_generate_tracking, clean_wandb, show_cmd):
    config = get_config(benchmark_config_file)

    global headers
    headers = [
        "benchmark_name",
        "system_name",
        "devices",
        "docker_name",
        "status",
        "cmd"
        ]

    if kill_all:
        print("============")
        print("Killing all existing dockers")
        print("============")
        subprocess.run("docker kill $(docker ps -q)", shell=True)
    
    if not run_only:
                    
        if not skip_generate_tracking:
            print("============")
            print("Generating benchmark tracking from config")
            print("============")
            
           
        if reset_tracking_db or not os.path.isfile(tracking_db):
            print(f"Resetting Tracking DB")
            if os.path.isfile(tracking_db):
               os.system(f"rm {tracking_db}")
            conn = create_connection(tracking_db)
            create_table(conn, """CREATE TABLE IF NOT EXISTS tasks (
index integer PRIMARY KEY,
benchmark_name text NOT NULL,
system_name text NOT NULL,
devices text NOT NULL,
docker_name text NOT NULL,
status text NOT NULL,
cmd text NOT NULL
);""")

        generate_all_benchmarks(config, interactive_mode, tracking_db)



    if clean_wandb:
        print("============")
        print("Cleaning wandb")
        print("============")
        os.environ['WANDB_API_KEY'] = config['wandb']['key']
        runs = wandb.Api().runs(f"{config['wandb']['user']}/{config['wandb']['project']}")

        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(clean_wandb_id, [(this_run.id, config) for this_run in runs])
        pool.close()

        if False in results:
            print("Cleaning wandb failed please check logs")

    if run or run_only or interactive_mode:
        print("============")
        print("Running benchmark")
        print("============")
        runner(tracking_db, interactive_mode, show_cmd)

    

        

if __name__ == '__main__':
    main()
