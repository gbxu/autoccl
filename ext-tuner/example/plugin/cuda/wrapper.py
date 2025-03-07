import ctypes
import sys
import os

class Pair(ctypes.Structure):
    _fields_ = [("key", ctypes.c_char_p),
                ("value", ctypes.c_int)]

WORKLOAD_SIZE = 3
CANDIDATE_SIZE = 10

class GroupInfo(ctypes.Structure):
    _fields_ = [("groupID", ctypes.c_int64),
                ("root", ctypes.c_int32),
                ("rank", ctypes.c_int32),
                ("nrank", ctypes.c_int32),
                ("nnode", ctypes.c_int32)]

class NCCLCandidateWrapper:
    def __init__(self, lib_path=None):
        try:
            # get current file directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # get dynamic library path
            lib_path = os.path.join(current_dir, "../../build/libwrapper.so")
            self.lib = ctypes.CDLL(lib_path)
            print("Library loaded successfully")
        except OSError as e:
            print(f"Failed to load library: {e}")
            sys.exit(1)

        self.lib.ncclGetValidCandidatesWrapper.argtypes = [
            ctypes.POINTER(GroupInfo), # pointer
            ctypes.POINTER(Pair), # pointer
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64), # pointer
            ctypes.c_bool,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), # pointer of pointer
            ctypes.POINTER(ctypes.c_size_t) # pointer
        ]
        self.lib.ncclGetValidCandidatesWrapper.restype = None

        self.lib.freeCandidates.argtypes = [ctypes.POINTER(ctypes.c_int32)]
        self.lib.freeCandidates.restype = None

    def nccl_get_valid_candidates(self, nrank, nnode, coll, size, tunerEnvs, scale2):
        pairs = [(key.encode('utf-8'), value) for key, value in tunerEnvs.items()]
        pairCount = len(pairs)
        pairPtr = (Pair * pairCount)(*pairs)

        group_info = GroupInfo(groupID=0, root=0, rank=0, nrank=nrank, nnode=nnode)

        workloadElemPtr = (ctypes.c_uint64 * WORKLOAD_SIZE)(*[0, coll, size])

        candidateElemPtr = ctypes.POINTER(ctypes.c_int32)() # allocate the pointer
        candidateElemCount = ctypes.c_size_t()

        self.lib.ncclGetValidCandidatesWrapper(
            ctypes.byref(group_info), # get the pointer
            pairPtr,
            pairCount,
            workloadElemPtr,
            scale2,
            ctypes.byref(candidateElemPtr), # pointer of pointer
            ctypes.byref(candidateElemCount) # get the pointer
        )

        candidates = []
        for i in range(int(candidateElemCount.value/CANDIDATE_SIZE)):
            candidate = []
            for j in range(CANDIDATE_SIZE):
                candidate.append(candidateElemPtr[CANDIDATE_SIZE*i+j])
            candidates.append(candidate)

        self.lib.freeCandidates(candidateElemPtr)

        return candidates
