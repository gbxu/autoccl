#
# See LICENSE.txt for license information
#
CUDA_HOME ?= /usr/local/cuda
TUNER_HOME ?= build
PLUGIN_NAME := nccl-plugin
TUNER_NAME := tuner
DEBUG ?= 0
SAFE ?= 0
NCCL_HOME ?= ../../build
#################################
INCLUDES := -I/usr/include -I/usr/local/include -I$(CUDA_HOME)/include -I./ -I./3rdparty/cereal/include

TUNER_SRC := src/tuner.cc
TUNER_OBJ_DIR := $(TUNER_HOME)/obj
TUNER_OBJ := $(TUNER_SRC:%.cc=$(TUNER_OBJ_DIR)/%.o)

# tuner
TUNER_SO := lib${TUNER_NAME}.so
# plugin for nccl...
PLUGIN_SO := lib${PLUGIN_NAME}.so
# executable files for testing
CUDA_BIN_LIST := $(patsubst test/%.cu,%,$(wildcard test/cuda/*.cu))
CC_BIN_LIST := $(patsubst test/%.cc,%,$(wildcard test/cuda/*.cc) )
CUDA_BINS := $(CUDA_BIN_LIST:%=${TUNER_HOME}/%)
CC_BINS := $(CC_BIN_LIST:%=${TUNER_HOME}/%)

#################################
NVCC = ${CUDA_HOME}/bin/nvcc
CXXFLAGS = -Wall -Wno-unused-function -Wextra -Wvla -Wconversion -Wshadow -Wnon-virtual-dtor -Wold-style-cast \
	-Weffc++ -std=c++14 # -Werror
NVCCFLAGS := -ccbin $(CXX) \
	$(NVCC_GENCODE) \
	--resource-usage \
	--expt-extended-lambda \
	-Xptxas -v \
	-Xfatbin -compress-all \
	-forward-unknown-to-host-compiler -Xcompiler ${CXXFLAGS}
CXX_LD = -ldl
NVCC_LD = -L${CUDA_HOME}/lib64 -lcudart
LD_NCCL = -L$(NCCL_HOME)/lib -lnccl
LD_TUNER = -L$(TUNER_HOME) -l${TUNER_NAME}

# Compiler flags for optimization or debugging
ifeq ($(DEBUG), 0)
CXXFLAGS  += -O3 -flto
NVCCFLAGS += -O3
else
CXXFLAGS  += -O0 -g -ggdb3 -fdiagnostics-color=always -DTUNER_TRACE
NVCCFLAGS += -O0 -G -g --display-error-number --generate-line-info --source-in-ptx
endif

##########  ##########

# Additional flags for safety checks if SAFE is not 0
SAFE_FLAGS := ASAN_OPTIONS=detect_odr_violation=0
ifneq ($(SAFE), 0)
# ASAN_LIB=$(shell clang -print-file-name=libclang_rt.asan-x86_64.so)
USE_THREAD_SANITIZER ?= 0
ifeq ($(USE_THREAD_SANITIZER), 1)
	CXXFLAGS  +=  -fsanitize=thread -DUSE_WORKERS_NCCLComm=1
	CXX_LD += -fsanitize=thread
	NVCCFLAGS += -fsanitize=thread
else
	CXXFLAGS  +=  -fsanitize=leak -fsanitize=address
	NVCCFLAGS +=  -fsanitize=leak -fsanitize=address
endif
CXXFLAGS  +=  -fstack-protector
CXXFLAGS  +=  -fsanitize=undefined
NVCCFLAGS +=  -fstack-protector
NVCCFLAGS +=  -fsanitize=undefined
endif

# Targets and dependencies
#################################

# Default target: Build TUNER_SO and PLUGIN_SO
default: $(TUNER_HOME)/$(TUNER_SO) $(TUNER_HOME)/$(PLUGIN_SO) $(TUNER_HOME)/libwrapper.so
	@echo "Finished " $^

# Test target: Build CUDA_BINS and CC_BINS
test: $(CUDA_BINS) $(CC_BINS)
	@echo "Finished " $^

# $^ => $@ : Link all dependencies into the TUNER_SO target
$(TUNER_HOME)/$(TUNER_SO): $(TUNER_OBJ)
	@printf "Compiling and Linking  %-35s > %s\n" $^ $@
	@mkdir -p $(TUNER_HOME)
	${CXX} ${CXXFLAGS} -fPIC -shared -Wl,-soname,$(TUNER_SO) ${INCLUDES} $^ -o $@ ${CXX_LD}

# Rule to compile .cc files => .o files
# $< => $@ : Link the first argument $< into the target $@
$(TUNER_OBJ_DIR)/%.o :  %.cc
	@printf "Compiling %-35s > %s\n" $< $@
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -fPIC $(INCLUDES) -c $< -o $@ || (echo "Failed to build $@"; exit 1)

# Rule to libplugin.so :  link tuner and nccl
$(TUNER_HOME)/$(PLUGIN_SO): plugin/cuda/plugin.cc $(TUNER_HOME)/$(TUNER_SO)
	@printf "Compiling and Linking  %-35s > %s\n" $< $@
	@mkdir -p $(TUNER_HOME)/
	${CXX} ${CXXFLAGS} -fPIC -shared -Wl,-soname,$(PLUGIN_SO) ${INCLUDES} -I$(NCCL_HOME)/include $< -o $@ ${CXX_LD} ${LD_TUNER}

$(TUNER_HOME)/libwrapper.so: plugin/cuda/wrapper.cc
	@printf "Compiling and Linking  %-35s > %s\n" $< $@
	@mkdir -p $(TUNER_HOME)/
	${CXX} ${CXXFLAGS} -fPIC -shared -Wl,-soname,libwrapper.so ${INCLUDES} $< -o $@ ${CXX_LD}

# Rule to test: link tuner to test
$(TUNER_HOME)/%: test/%.cu $(TUNER_HOME)/$(TUNER_SO)
	@printf "Compiling and Linking  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	${NVCC} ${NVCCFLAGS} -no-pie ${INCLUDES} $< -x cu -o $@ ${NVCC_LD} ${LD_TUNER}

# Rule to loader: link nccl to test plugin loading
$(TUNER_HOME)/cuda/test_loader: test/cuda/test_loader.cc $(TUNER_HOME)/$(PLUGIN_SO)
	@printf "Compiling and Linking  %-35s > %s\n" $< $@
	@mkdir -p $(dir $@)
	${CXX} ${CXXFLAGS} -no-pie ${INCLUDES} -I$(NCCL_HOME)/include $< -o $@ ${CXX_LD} ${LD_NCCL}

#################################
.PHONY: run perf clean check_nccl
run:
	@echo "USE: make clean && make test DEBUG=1 SAFE=1 && make run"
	@echo "USE: make clean && make test DEBUG=1 && make run"
	@mkdir -p $(TUNER_HOME)/make_test_log
	for bin in $(CUDA_BINS) $(CC_BINS); do \
		echo "Testing $$bin"; \
		$(SAFE_FLAGS) LD_LIBRARY_PATH=$(TUNER_HOME):$(NCCL_HOME)/lib:${LD_LIBRARY_PATH} NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL TUNER_DUMP_DATABASE=$(TUNER_HOME)/make_test_log TUNER_VERBOSE=TRACE NCCL_TUNER_PLUGIN="$(TUNER_HOME)/${PLUGIN_SO}" ./$$bin 15; \
	done

perf:
	@echo "Perf Debuging..."
	@echo "USE: make clean && make test DEBUG=1 && make perf"
	@for bin in $(CUDA_BINS) $(CC_BINS); do \
		echo "Testing $$bin"; \
		mkdir -p $$bin"_log"; \
		LD_LIBRARY_PATH=$(TUNER_HOME):$(NCCL_HOME)/lib:${LD_LIBRARY_PATH} NCCL_DEBUG=VERSION TUNER_DUMP_DATABASE=$$bin"_log"/ TUNER_VERBOSE=ERROR TUNER_MODE=0 NCCL_TUNER_PLUGIN="$(TUNER_HOME)/${PLUGIN_SO}" perf record -g -o $$bin"_log"/perf.data ./$$bin 1; \
	done

clean:
	@echo "Remove..."
	rm -rf $(TUNER_HOME)/
	rm -rf $(TUNER_HOME)/make_test_log

# check NCCL_HOME
check_nccl:
	@if [ -d "$(NCCL_HOME)" ]; then \
		echo "NCCL_HOME path exists: $(NCCL_HOME)"; \
	else \
		echo "NCCL_HOME path does not exist: $(NCCL_HOME)"; \
		exit 1; \
	fi

#################################
