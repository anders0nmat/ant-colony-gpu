# Usage
# >> make <platform>
# or
# >> make <platform>-release
#
# Supported platforms: linux, mac

.PHONY: help
help:
	@echo "Usage:"
	@echo ">> make <platform>"
	@echo ">> make <platform>-release"
	@echo ""
	@echo "Supported platforms: linux, mac, windows"

CXX_COMPILER := clang++
CXX_VERSION := c++17
CXX_WARNINGS := -Wall -Wpedantic -Werror

LOCATION_LIBRARIES := lib/
LOCATION_INCLUDES := include/
LOCATION_CPP := src/*.cpp #src/variants/*.cpp
LOCATION_OUTPUT := ./build/main

ETC_FLAGS := #-DGUI

MAC_FLAGS := #-lglfw3-mac -framework Cocoa -framework OpenGL -framework IOKit
LINUX_FLAGS := -lOpenCL#-lglfw3-linux -lGL -lX11
WINDOWS_FLAGS := 

DEBUG = --debug
RELEASE = -O3

# Used for execution, do not touch
FLAGS =
OPTS =


.PHONY: linux
linux: linux-debug

.PHONY: linux-debug
linux-debug: FLAGS = $(LINUX_FLAGS)
linux-debug: debug

.PHONY: linux-release
linux-release: FLAGS = $(LINUX_FLAGS)
linux-release: release

.PHONY: mac
mac: mac-debug

.PHONY: mac-debug
mac-debug: FLAGS = $(MAC_FLAGS)
mac-debug: debug

.PHONY: mac-release
mac-release: FLAGS = $(MAC_FLAGS)
mac-release: release

.PHONY: windows
windows: windows-debug

.PHONY: windows-debug
windows-debug: FLAGS = $(WINDOWS_FLAGS)
windows-debug: debug

.PHONY: windows-release
windows-release: FLAGS = $(WINDOWS_FLAGS)
windows-release: release



.PHONY: debug
debug: OPTS = $(DEBUG)
debug: executable

.PHONY: release
release: OPTS = $(RELEASE)
release: executable


.PHONY: executable
executable:
	rm -If $(LOCATION_OUTPUT)
	$(CXX_COMPILER) $(LOCATION_CPP) -o $(LOCATION_OUTPUT) -std=$(CXX_VERSION) $(CXX_WARNINGS) -L $(LOCATION_LIBRARIES) -I $(LOCATION_INCLUDES) $(OPTS) $(FLAGS) $(ETC_FLAGS)

