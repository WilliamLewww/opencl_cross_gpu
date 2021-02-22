#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>

#include <CL/cl.h>

int main(void) {
  cl_uint platformCount = 0;
  cl_platform_id platformID;
  clGetPlatformIDs(1, &platformID, &platformCount);

  cl_device_id deviceID;
  cl_uint deviceCount = 0;
  clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &deviceCount);

  char* deviceName;
  clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 100, deviceName, NULL);
  printf("%s\n", deviceName);

  return 0;
}
