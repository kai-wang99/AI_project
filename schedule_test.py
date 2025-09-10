# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:27:20 2025

@author: k_wan
"""

import torch
import torch.utils.cpp_extension
import os

# based on Jeremy's Lecture 3 notebook
cuda_begin = r'''
#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''

cuda_src = cuda_begin + r'''
__global__ void rgb_to_grayscale_kernel(unsigned char* out, unsigned char* in, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = 0.2989f*in[i] + 0.5870f*in[i+n] + 0.1140f*in[i+2*n];  // fix with f found by Andreas...
}

torch::Tensor rgb_to_grayscale_out(torch::Tensor output, const torch::Tensor& input) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    TORCH_CHECK((h == output.size(0)) || (w == output.size(1)) || (output.device() == input.device())
                || (output.scalar_type() == input.scalar_type()));
    int threads = 256;
    rgb_to_grayscale_kernel<<<cdiv(w*h,threads), threads>>>(
        output.data_ptr<unsigned char>(), input.data_ptr<unsigned char>(), w*h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

torch::Tensor rgb_to_grayscale(const torch::Tensor& input) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    auto output = torch::empty({h,w}, input.options());
    rgb_to_grayscale_out(output, input);
    return output;
}
'''

cpp_src = """
torch::Tensor rgb_to_grayscale(const torch::Tensor& input);
torch::Tensor rgb_to_grayscale_out(torch::Tensor outpuit, const torch::Tensor& input);
"""

#import os
#os.environ['CXX'] = '/usr/lib/ccache/g++-11'
#os.environ['CC'] = '/usr/lib/ccache/gcc-11'

module = torch.utils.cpp_extension.load_inline(
    "test_ext", cpp_src, cuda_src, 
    functions=['rgb_to_grayscale', 'rgb_to_grayscale_out'], extra_cuda_cflags=['--ptxas-options=-v'], verbose=True)



n = 2048
t = torch.randint(0, 256, (3, n, n), dtype=torch.uint8, device="cuda")
out = module.rgb_to_grayscale(t); torch.cuda.synchronize()

import time
t0 = time.perf_counter_ns()
for i in range(10_000):
    module.rgb_to_grayscale_out(out, t)
torch.cuda.synchronize()
t1 = time.perf_counter_ns()

print((t1-t0) / 10_000 / 1_000, "µs") 


with torch.profiler.profile(on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/schedule'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True) as prof:
    for i in range(10_000):
        module.rgb_to_grayscale_out(out, t)
        torch.cuda.synchronize()

print(prof.key_averages().table())