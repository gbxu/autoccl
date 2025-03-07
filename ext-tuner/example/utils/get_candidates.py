import sys
import os
import time
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
wrapper_dir = os.path.join(current_dir, "../plugin/cuda/")
sys.path.append(wrapper_dir)
from wrapper import NCCLCandidateWrapper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrank", type=int, default=8)
    parser.add_argument("--nnode", type=int, default=1)
    parser.add_argument("--coll", type=str, default="AllReduce")
    parser.add_argument("--size", type=int, default=1024, help="nbyte per rank")
    parser.add_argument("--expire", type=int, default=10, help="expire per candidate, -1 is forever")
    parser.add_argument("--scale2", action='store_true', default=False, help="")
    args = parser.parse_args()
    coll_name = f"nnode{args.nnode}_nrank{args.nrank}_coll{args.coll}_size{args.size}"
    coll_to_key = {
        "Broadcast" : 0,
        "Reduce" : 1,
        "AllGather" : 2,
        "ReduceScatter" : 3,
        "AllReduce" : 4,
        "SendRecv" : 5,
        "All2All" : 9
    }

    wrapper = NCCLCandidateWrapper()
    # p2plevel=7 by default in NCCL
    # gdrlevel=4 by default in NCCL
    map_dict = {
        "tuner_extraP2PCE": 0,
        "tuner_extraSHM": 1,
        "tuner_p2pLevel": -1,
        # set tuner_p2pChunkSize=512*1024 for nvlink-1-node
        "tuner_p2pChunkSize": 128*1024,
        "tuner_p2pnChannelsPerPeer": 128,
        "tuner_p2pnChannels": 128,
        "tuner_nChannels": 128,
        "tuner_treeupdown_allreduce_simple": 0
    }
    candidates = wrapper.nccl_get_valid_candidates(
        nrank=args.nrank,
        nnode=args.nnode,
        coll=coll_to_key[args.coll],
        size=args.size,
        tunerEnvs=map_dict,
        scale2=args.scale2)

    count_after_filter = 0
    file = open(f"{coll_name}.txt", 'w')
    print(f"writting {coll_name}.txt")
    last_output = None
    for candidate in candidates:
        # you can filter the candidate here
        algo, proto, copytype, p2plevel, nc, nt, chunksize, _, _, _ = candidate
        '''example
        if algo != 1:
            continue
        if proto != 2:
            continue
        if nc & (nc - 1) != 0:
            continue
        if nt & (nt - 1) != 0:
            continue
        if chunksize & (chunksize - 1) != 0:
            continue
        '''
        count_after_filter += 1
        specific_candidate = f"{args.nrank}; -1 {coll_to_key[args.coll]} {args.size};{' '.join(map(str, candidate))}"
        output = f"{specific_candidate};{args.expire}"
        last_output = f"{specific_candidate};-1"
        file.write(output + '\n')
    file.write(last_output + '\n')
    file.close()

    print(f"The communication: nrank={args.nrank}, nnode={args.nnode}, coll={args.coll}, sizePerRank={args.size},\n"
        f"tunerEnvs={map_dict},\n"
        f"scale2={args.scale2}, has {len(candidates)} candidates, after filter: {count_after_filter} candidates.")

    print(f"set environment variable TUNER_PROFILE_MORE={count_after_filter*args.expire},\n"
        f"then call {count_after_filter*args.expire} times {coll_name},\n"
        f"add more repeats will be safer.")
