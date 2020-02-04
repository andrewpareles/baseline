# Copyright (c) 2019, University of Washington All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# Redistributions of source code must retain the above copyright notice, this list
# of conditions and the following disclaimer.
# 
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
# 
# Neither the name of the copyright holder nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

################################################################################
# Paths / Environment Configuration
################################################################################
_REPO_ROOT ?= $(shell git rev-parse --show-toplevel)

-include $(_REPO_ROOT)/environment.mk

# These variables are used by simlibs.mk
TESTBENCH_PATH := $(BSG_F1_DIR)/testbenches
LIBRARIES_PATH := $(BSG_F1_DIR)/libraries
HARDWARE_PATH  := $(BSG_F1_DIR)/hardware
# CL_DIR Means "Custom Logic" Directory, and is an AWS-FPGA holdover. cadenv.mk
# checks that it is set
CL_DIR         := $(BSG_F1_DIR)

# The following makefile fragment verifies that the tools and CAD environment is
# configured correctly.
-include $(BSG_F1_DIR)/cadenv.mk

################################################################################
# Simulation Libraries (C/C++ AND Verilog)
################################################################################
# PROJECT defines the Verilog Simulation Library Target
PROJECT        := baseline

# simlibs.mk defines build rules for hardware and software simulation libraries
# that are necessary for running cosimulation. These are dependencies for
# regression since running $(MAKE) recursively does not prevent parallel builds
# of identical rules -- which causes errors.
-include $(TESTBENCH_PATH)/simlibs.mk

# libbsg_manycore_runtime will be compiled in $(LIBRARIES_PATH)
LDFLAGS        += -lbsg_manycore_runtime -lm
LDFLAGS        += -L$(LIBRARIES_PATH) -Wl,-rpath=$(LIBRARIES_PATH)

VCS_LDFLAGS    += $(foreach def,$(LDFLAGS),-LDFLAGS "$(def)")

VCS_VFLAGS     += -M +lint=TFIPC-L -ntb_opts tb_timescale=1ps/1ps -lca -v2005
VCS_VFLAGS     += -timescale=1ps/1ps -sverilog -full64 -licqueue
VCS_VFLAGS     += -debug_pp
VCS_VFLAGS     += +memcbk

INCLUDES       += -I$(LIBRARIES_PATH) -I$(VCS_HOME)/linux64/lib/

CCPPDEFINES    += -DCOSIM -DVCS
CXXDEFINES     += $(CCPPDEFINES)
CDEFINES       += $(CCPPDEFINES)

CFLAGS         += -std=c99 $(CDEFINES) $(INCLUDES)
CXXFLAGS       += -std=c++11 -lstdc++ $(CXXDEFINES) $(INCLUDES)

# HOST_OBJECTS defines the object files that that are linked as part of
# the kernel. It is derived from HOST_*SOURCES (see below) but other
# objects can be added and linked as necessary.
HOST_OBJECTS   += $(HOST_SSOURCES:.s=.o)
HOST_OBJECTS   += $(HOST_CSOURCES:.c=.o)
HOST_OBJECTS   += $(HOST_CXXSOURCES:.cpp=.o)

$(HOST_TARGET):
	$(error $(shell echo -e "$(RED)Native host compilation not \
				implemented. Run \`make $(HOST_TARGET).cosim\`\
				to build the cosimulation binary$(NC)"))

# VCS Generates an executable file by linking against the .o files in
# $(HOST_OBJECTS). WRAPPER_NAME is the top-level simulation wrapper
# defined in simlibs.mk

# We parallelize VCS compilation, but we leave a few cores on the table.
$(HOST_TARGET).cosim: NPROCS := $(shell echo "(`nproc`/4 + 1)" | bc)
$(HOST_TARGET).cosim: $(HOST_OBJECTS) $(SIMLIBS)
	SYNOPSYS_SIM_SETUP=$(TESTBENCH_PATH)/synopsys_sim.setup \
	vcs tb glbl -j$(NPROCS) $(WRAPPER_NAME) $(filter %.o,$^) \
		-Mdirectory=$@.tmp \
		$(VCS_LDFLAGS) $(VCS_VFLAGS) -o $@ -l $@.vcs.log

host.compile.clean: 
	rm -rf $(HOST_OBJECTS)

host.link.clean:
	rm -rf $(HOST_TARGET)

cosim.clean: host.link.clean host.compile.clean
	rm -rf *.cosim{.daidir,.tmp,} 64
	rm -rf vc_hdrs.h ucli.key
	rm -rf *.vpd *.vcs.log