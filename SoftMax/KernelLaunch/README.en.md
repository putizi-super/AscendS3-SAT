## Overview
This example introduces the direct kernel function invocation method for the MoeSoftMaxTopk operator.

## Directory Structure Introduction
``` 
├── KernelLaunch                      // Invokes the MoeSoftMaxTopk custom operator using direct kernel function invocation
│   └── MoeSoftMaxTopkKernelInvocation     // Example of invoking the kernel function using the Kernel Launch method
``` 

## Compiling and Running the Example Operator
For custom operator projects, the compilation and running process includes the following steps:
- Compile the custom operator project;
- Invoke and execute the custom operator;

Detailed operations are as follows.

### 1. Obtain the Source Code Package
Before compiling and running this example, please refer to [Preparation: Obtain the Example Code](../README.en.md#codeready) to obtain the source code package.

### 2. Compile and Run the Example Project
- [Run the MoeSoftMaxTopkKernelInvocation Example](./MoeSoftMaxTopkKernelInvocation/README.en.md)

## Update Log
  | Date       | Update Items | Notes |
  |------------|--------------|-------|
  | 2024/07/02 | Updated readme structure | Requires running on community CANN package version 8.0.RC1.alpha003 or later |