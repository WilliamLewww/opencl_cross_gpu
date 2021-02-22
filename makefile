OPENCL_SDK_PATH = /usr/local/OpenCL-SDK

EXEC = main.out

SRCS := $(wildcard src/*.c)
OBJS := $(notdir $(SRCS:%.c=%.o))

CFLAGS =-I$(OPENCL_SDK_PATH)/include/api
LDFLAGS =-L$(OPENCL_SDK_PATH)/build -lOpenCL

$EXEC: $(OBJS)
	gcc $(CFLAGS) -o bin/$(EXEC) build/*.o $(LDFLAGS)

%.o: src/%.c
	gcc $(CFLAGS) -c $^ -o build/$@
