import torch
import torch.distributed as dist
import argparse
import os
import datetime
import sys
import numpy as np

def profiling(args, nBytes, repeat):
    data_type = torch.float32
    data_type_size = torch.finfo(data_type).bits // 8

    if repeat == 0 or nBytes//data_type_size == 0:
        return
    comm_begin_event = torch.cuda.Event(enable_timing=True)
    comm_end_event = torch.cuda.Event(enable_timing=True)

    tensor = torch.randn([1, nBytes//data_type_size], dtype=data_type).to(args.local_rank)
    comm_begin_event.record()
    for i in range(repeat):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    comm_end_event.record()
    comm_end_event.synchronize() # blocking CPU thread until the event completes.
    comm_gpu_time = comm_begin_event.elapsed_time(comm_end_event)/repeat
    GB_s = nBytes*1.0/1024/1024/1024/(comm_gpu_time/1000)
    if args.rank == 0:
        print(f"avged by repeat: {repeat:>5}, bytes={nBytes:>10}, nrank={args.world_size:>4}: {GB_s:>5.2f} GB/s")

    comm_begin_event.record()
    for i in range(100):
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    comm_end_event.record()
    comm_end_event.synchronize() # blocking CPU thread until the event completes.
    comm_gpu_time = comm_begin_event.elapsed_time(comm_end_event)/100
    GB_s = nBytes*1.0/1024/1024/1024/(comm_gpu_time/1000)
    if args.rank == 0:
        print(f"after pretuning, avged by repeat 100, bytes={nBytes:>10}, nrank={args.world_size:>4}: {GB_s:>5.2f} GB/s")

    torch.cuda.synchronize(args.local_rank)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0, help="global rank")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--world_size", type=int, default=1, help="world size")
    parser.add_argument("--master_ip", type=str, default="localhost", help="master ip")
    parser.add_argument("--master_port", type=str, default="6000", help="master port")

    args = parser.parse_args()
    if os.getenv('OMPI_COMM_WORLD_SIZE') is not None: # for mpirun
        args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', args.rank))
        args.local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', args.local_rank))
        args.world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', args.world_size))
        os.environ['MASTER_ADDR'] = str(args.master_ip)
        os.environ['MASTER_PORT'] = str(args.master_port)
    elif os.getenv('WORLD_SIZE') is not None: # for torchrun
        args.rank = int(os.environ.get('RANK', args.rank))
        args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        args.world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
        args.master_ip = str(os.environ.get('MASTER_ADDR', args.master_ip))
        args.master_port = str(os.environ.get('MASTER_PORT', args.master_port))

    torch.cuda.set_device(args.local_rank)

    tuning_steps = 5 # for warmup
    tuning_steps += 5 # for native
    tuning_steps += int(os.environ.get('TUNER_PRETRAIN_STEPS'))
    tuning_steps += int(os.environ.get('TUNER_TRAIN_STEPS'))
    # to avoid the remaing round
    tuning_steps += 5 # in case

    if args.rank == 0:
        os.environ['TUNER_ROLE'] = "COORDINATOR"

    if args.world_size > 1:
        init_method = 'tcp://'
        init_method += args.master_ip + ':' + args.master_port
        torch.distributed.init_process_group(backend='nccl',
                                            rank=args.rank,
                                            world_size=args.world_size,
                                            init_method=init_method,
                                            timeout=datetime.timedelta(seconds=14400000))

    for nBytes in [1024*(2**i) for i in range(19)]:
        profiling(args, nBytes, tuning_steps)
