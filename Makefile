#
# Copyright 2019-2020 Xilinx, Inc.
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
# makefile-generator v1.0.3
#

############################## Help Section ##############################
.PHONY: help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to generate the design for specified Target and Shell."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make clean "
	$(ECHO) "      Command to remove the generated non-hardware files."
	$(ECHO) ""
	$(ECHO) "  make cleanall"
	$(ECHO) "      Command to remove all the generated files."
	$(ECHO) ""
	$(ECHO)  "  make test DEVICE=<FPGA platform>"
	$(ECHO)  "     Command to run the application. This is same as 'run' target but does not have any makefile dependency."
	$(ECHO)  ""
	$(ECHO) "  make sd_card TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to prepare sd_card files."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make run TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to run application in emulation."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make build TARGET=<sw_emu/hw_emu/hw> DEVICE=<FPGA platform> HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build xclbin application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""
	$(ECHO) "  make host HOST_ARCH=<aarch32/aarch64/x86> EDGE_COMMON_SW=<rootfs and kernel image path>"
	$(ECHO) "      Command to build host application."
	$(ECHO) "      By default, HOST_ARCH=x86. HOST_ARCH and EDGE_COMMON_SW is required for SoC shells"
	$(ECHO) ""

############################## Setting up Project Variables ##############################
# Points to top directory of Git repository
MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
COMMON_REPO ?= $(realpath $(dir $(firstword $(MAKEFILE_LIST))))
PWD = $(shell readlink -f .)
XF_PROJ_ROOT = $(shell readlink -f $(COMMON_REPO))

# Define common parameters
PARAM_DEFS := -D H_IN=784 -D M_IN=2 -D H_HID=32 -D M_HID=128 -D H_UT=1 -D M_UT=10 -D NACTHI=64 -D NSILHI=64

TARGET := hw
HOST_ARCH := aarch64
SYSROOT := 

HOST_COMPILE := mnistmain_FPGA

HOST_MAIN_SRC := ./test/MNIST_ZCU104/$(HOST_COMPILE).cpp

include ./utils.mk

KERNEL_COMPILE := BCPNN_Kernel

XSA := 
ifneq ($(DEVICE), )
XSA := $(call device2xsa, $(DEVICE))
endif
TEMP_DIR := ./_x.$(TARGET).$(XSA)
BUILD_DIR := ./build_dir.$(TARGET).$(XSA)

# SoC variables
RUN_APP_SCRIPT = ./run_app.sh
PACKAGE_OUT = ./package.$(TARGET)

LAUNCH_EMULATOR = $(PACKAGE_OUT)/launch_$(TARGET).sh
RESULT_STRING = TEST PASSED

VPP := v++
CMD_ARGS = $(BUILD_DIR)/$(KERNEL_COMPILE).xclbin
SDCARD := sd_card

include $(XF_PROJ_ROOT)/common/includes/opencl/opencl.mk
CXXFLAGS += $(opencl_CXXFLAGS) -Wall -O3 -g -std=c++17 `pkg-config --cflags opencv4` -pthread -lgdal -llapack -lblas
LDFLAGS += $(opencl_LDFLAGS) `pkg-config --libs opencv4` -pthread -lgdal -llapack -lblas


ifeq ($(findstring nodma, $(DEVICE)), nodma)
$(error [ERROR]: This example is not supported for $(DEVICE).)
endif

############################## Setting up Host Variables ##############################
#Include Required Host Source Files
CXXFLAGS += -I$(XF_PROJ_ROOT)/common/includes/xcl2 
CXXFLAGS += -I$(XF_PROJ_ROOT)/HLS_library/include
HOST_SRCS += $(XF_PROJ_ROOT)/common/includes/xcl2/xcl2.cpp $(HOST_MAIN_SRC)
# Host compiler global settings
CXXFLAGS += -fmessage-length=0 -DHLS_NO_XIL_FPO_LIB
LDFLAGS += -lrt -lstdc++ 

LIB_SRCDIR := ./libsrc/src

#ifneq ($(HOST_ARCH), x86)
#	LDFLAGS += --sysroot=$(SYSROOT)
#endif

ifneq ($(HOST_ARCH), x86)
  ifneq ($(SYSROOT),)
    LDFLAGS += --sysroot=$(SYSROOT)
  endif
endif

# List the corresponding object files (all placed in ./bin)
HOST_OBJS := ./bin/Parseparam.o \
             ./bin/Pop.o \
             ./bin/PatternFactory.o \
             ./bin/Logger.o \
             ./bin/AxoDelay.o \
             ./bin/Globals.o \
             ./bin/Netw.o \
             ./bin/Prj.o \
             ./bin/Analys.o \
             ./bin/Probe.o 

CCFLAGS := -std=c++17 -O3 -g -I./libsrc/include/ -I$(XF_PROJ_ROOT)/HLS_library/include -DHLS_NO_XIL_FPO_LIB
CCFLAGS += $(PARAM_DEFS)

CXXFLAGS += $(PARAM_DEFS)

############################## Setting up Kernel Variables ##############################
# Kernel compiler global settings
VPP_FLAGS += -t $(TARGET) --platform $(DEVICE) --save-temps 
ifneq ($(TARGET), hw)
	VPP_FLAGS += -g
endif
VPP_FLAGS_CONFIG +=  --config ./configure.cfg

VPP_FLAGS_HLS := --kernel_frequency 100

VPP_FLAGS += $(PARAM_DEFS)

EXECUTABLE = ./$(HOST_COMPILE)
EMCONFIG_DIR = $(TEMP_DIR)
EMU_DIR = $(SDCARD)/data/emulation

############################## Declaring Binary Containers ##############################
BINARY_CONTAINERS += $(BUILD_DIR)/$(KERNEL_COMPILE).xclbin
BINARY_CONTAINER_adder_OBJS += $(TEMP_DIR)/$(KERNEL_COMPILE).xo

############################## Setting Targets ##############################
CP = cp -rf

.PHONY: all clean cleanall docs emconfig
all: check-devices $(EXECUTABLE) $(BINARY_CONTAINERS) emconfig sd_card

.PHONY: host
host: $(EXECUTABLE)

.PHONY: build
build: check-vitis $(BINARY_CONTAINERS)

.PHONY: xclbin
xclbin: build

############################## Setting Rules for Binary Containers (Building Kernels) ##############################
$(TEMP_DIR)/$(KERNEL_COMPILE).xo: libsrc/src/$(KERNEL_COMPILE).cpp
	mkdir -p $(TEMP_DIR)
	$(VPP) $(VPP_FLAGS) $(VPP_FLAGS_CONFIG) $(VPP_FLAGS_HLS) -c -k $(KERNEL_COMPILE) --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' '$<' --hls.pre_tcl ./compile_hls.tcl -I ./libsrc/include/
$(BUILD_DIR)/$(KERNEL_COMPILE).xclbin: $(BINARY_CONTAINER_adder_OBJS)
	mkdir -p $(BUILD_DIR)
ifeq ($(HOST_ARCH), x86)
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(BUILD_DIR)  -o'$(BUILD_DIR)/$(KERNEL_COMPILE).link.xclbin' $(+) -I ./libsrc/include/
	$(VPP) -p $(BUILD_DIR)/$(KERNEL_COMPILE).link.xclbin -t $(TARGET) --platform $(DEVICE) --package.out_dir $(PACKAGE_OUT) -o $(BUILD_DIR)/$(KERNEL_COMPILE).xclbin -I ./libsrc/include/
else
	$(VPP) $(VPP_FLAGS) -l $(VPP_LDFLAGS) --temp_dir $(BUILD_DIR) -o'$(BUILD_DIR)/$(KERNEL_COMPILE).xclbin' $(+) -I ./libsrc/include/
endif

############################## Setting Rules for Host (Building Host Executable) ##############################
$(EXECUTABLE): $(HOST_SRCS) | $(HOST_OBJS)
		$(CXX) $(HOST_OBJS) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) -I ./libsrc/include/

# Pattern rule for compiling the common library sources from LIB_SRCDIR.
./bin/%.o: $(LIB_SRCDIR)/%.cpp
	@mkdir -p ./bin
	$(CXX) $(CCFLAGS) -c $< -o $@

emconfig:$(EMCONFIG_DIR)/emconfig.json
$(EMCONFIG_DIR)/emconfig.json:
	emconfigutil --platform $(DEVICE) --od $(EMCONFIG_DIR)

############################## Setting Essential Checks and Running Rules ##############################
run: all
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	$(CP) $(EMCONFIG_DIR)/emconfig.json .
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(LAUNCH_EMULATOR_CMD)
endif
else
ifeq ($(HOST_ARCH), x86)
	$(EXECUTABLE) $(CMD_ARGS)
endif
endif


.PHONY: test
test: $(EXECUTABLE)
ifeq ($(TARGET),$(filter $(TARGET),sw_emu hw_emu))
ifeq ($(HOST_ARCH), x86)
	XCL_EMULATION_MODE=$(TARGET) $(EXECUTABLE) $(CMD_ARGS)
else
	$(LAUNCH_EMULATOR_CMD)
endif
else
ifeq ($(HOST_ARCH), x86)
	$(EXECUTABLE) $(CMD_ARGS)
else
	$(ECHO) "Please copy the content of sd_card folder and data to an SD Card and run on the board"
endif
endif


############################## Preparing sdcard ##############################
sd_card: $(BINARY_CONTAINERS) $(EXECUTABLE) gen_run_app
ifneq ($(HOST_ARCH), x86)
	$(VPP) -p $(BUILD_DIR)/$(KERNEL_COMPILE).xclbin -t $(TARGET) --platform $(DEVICE) --package.out_dir $(PACKAGE_OUT) --package.rootfs $(EDGE_COMMON_SW)/rootfs.ext4 --package.sd_file $(SD_IMAGE_FILE) --package.sd_file xrt.ini --package.sd_file $(RUN_APP_SCRIPT) --package.sd_file $(EXECUTABLE) -o $(KERNEL_COMPILE).xclbin
endif

############################## Cleaning Rules ##############################
# Cleaning stuff
clean:
	-$(RMDIR) $(EXECUTABLE) $(XCLBIN)/{*sw_emu*,*hw_emu*} 
	-$(RMDIR) profile_* TempConfig system_estimate.xtxt *.rpt *.csv 
	-$(RMDIR) src/*.ll *v++* .Xil emconfig.json dltmp* xmltmp* *.log *.jou *.wcfg *.wdb
	-$(RMDIR) bin/*.o

cleanall: clean
	-$(RMDIR) build_dir* sd_card*
	-$(RMDIR) package.*
	-$(RMDIR) _x* *xclbin.run_summary qemu-memory-_* emulation _vimage pl* start_simulation.sh *.xclbin

