#define CL_TARGET_OPENCL_VERSION 300

#include <stdio.h>

#include <CL/cl.h>

void printKernelBuildLog(cl_device_id deviceID, cl_program program) {
  char buffer[4096];
  size_t length;
  clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
  printf("%s\n",buffer);
}

int main(void) {
  cl_uint platformCount = 0;
  cl_platform_id platformID;
  clGetPlatformIDs(1, &platformID, &platformCount);

  cl_device_id deviceID;
  cl_uint deviceCount = 0;
  clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &deviceCount);

  char* deviceName = (char*)malloc(255);
  clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 255, deviceName, NULL);
  printf("%s\n", deviceName);
  free(deviceName);

  cl_int error;
  cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformID, 0 };
  cl_context context = clCreateContext(properties, 1, &deviceID, NULL, NULL, &error);

  cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceID, NULL, &error);

  FILE* kernelFile = fopen("src/kernels/basic.kernel", "rb");
  fseek(kernelFile, 0, SEEK_END);
  uint32_t kernelFileSize = ftell(kernelFile);
  fseek(kernelFile, 0, SEEK_SET);

  char* kernelFileBuffer = (char*)malloc(sizeof(char*) * kernelFileSize);
  fread(kernelFileBuffer, 1, kernelFileSize, kernelFile);
  fclose(kernelFile);

  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelFileBuffer, NULL, &error);
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  printKernelBuildLog(deviceID, program);

  cl_kernel kernel = clCreateKernel(program, "squareKernel", &error);

  float inputHost[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  float outputHost[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  cl_mem inputDevice = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * 10, NULL, NULL);
  cl_mem outputDevice = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * 10, NULL, NULL);
  clEnqueueWriteBuffer(commandQueue, inputDevice, CL_TRUE, 0, sizeof(float) * 10, inputHost, 0, NULL, NULL);

  const uint64_t globalSize = 10;
  const uint64_t blockSize = 1;

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputDevice);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputDevice);
  clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalSize, &blockSize, 0, NULL, NULL);
  clFinish(commandQueue);

  clEnqueueReadBuffer(commandQueue, outputDevice, CL_TRUE, 0, sizeof(float) * 10, outputHost, 0, NULL, NULL);

  for (int x = 0; x < 10; x++) {
    printf("%f\n", outputHost[x]);
  }

  clReleaseMemObject(outputDevice);
  clReleaseMemObject(inputDevice);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(commandQueue);
  clReleaseContext(context);

  return 0;
}
