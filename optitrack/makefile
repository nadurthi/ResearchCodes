include = -I../../include
libpath = -L../../lib
libs = -lNatNetLibShared

all: build/optitrack_recorder

build/SampleClient: 
	g++ optitrack_recorder.cpp $(include) $(libpath) $(libs) -o build/optitrack_recorder

.PHONY: clean
clean:
	@rm -f ./build/optitrack_recorder