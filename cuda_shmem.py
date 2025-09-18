import os,math,sys,torch,re,numpy as np
from types import SimpleNamespace as ns
from collections import namedtuple

from utils import cuda_begin,cdiv
from torch.utils.cpp_extension import load_inline

import time

dim3 = namedtuple('dim3', ['x','y','z'], defaults=(1,1))

torch.manual_seed(42);
m1 = torch.rand(5120, 256)
m1s = m1[:4]
m2 = torch.rand(256,5120)
m2s = m2[:,:4]

def blk_kernel2d(f, blocks, threads, *args):
    for i0 in range(blocks.y):
        for i1 in range(blocks.x):
            for j0 in range(threads.y):
                for j1 in range(threads.x): f(dim3(i1,i0), dim3(j1,j0), threads, *args)
     
def matmul_bk(blockIdx, threadIdx, blockDim, m, n, out, h, w, k):
    r = blockIdx.y*blockDim.y + threadIdx.y
    c = blockIdx.x*blockDim.x + threadIdx.x
    
    if (r>=h or c>=w): return
    o = 0.
    for i in range(k): o += m[r*k+i] * n[i*w+c]
    out[r*w+c] = o

def matmul_2d(m, n):
    h,k  = m.shape
    k2,w = n.shape
    assert k==k2, "Size mismatch!"
    output = torch.zeros(h, w, dtype=m.dtype)
    tpb = dim3(16,16)
    blocks = dim3(cdiv(w,tpb.x), cdiv(h,tpb.y))
    blk_kernel2d(matmul_bk, blocks, tpb,
                 m.flatten(), n.flatten(), output.flatten(), h, w, k)
    return output
     
print(torch.isclose(matmul_2d(m1s, m2s), m1s@m2s).all())
        
#CUDA

cuda_src = cuda_begin + r'''
__global__ void matmul_k(float* m, float* n, float* out, int h, int w, int k) {
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    int c = blockIdx.x*blockDim.x + threadIdx.x;

    if (r>=h || c>=w) return;
    float o = 0;
    for (int i = 0; i<k; ++i) o += m[r*k+i] * n[i*w+c];
    out[r*w+c] = o;
}

torch::Tensor matmul(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h = m.size(0);
    int w = n.size(1);
    int k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch!");
    auto output = torch::zeros({h, w}, m.options());

    dim3 tpb(16,16);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    matmul_k<<<blocks, tpb>>>(
        m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
'''

fname = 'matmul'

def get_sig(fname, src):
    res = re.findall(rf'^(.+\s+{fname}.*?)\s*{{?\s*$', src, re.MULTILINE)
    return res[0]+';' if res else None

def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                       extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")
     
#cpp_src = get_sig(fname, cuda_src)
cpp_src = "torch::Tensor matmul(torch::Tensor m, torch::Tensor n);"

module = load_cuda(cuda_src, cpp_src, [fname])
m1c,m2c = m1.contiguous().cuda(),m2.contiguous().cuda()
module.matmul(m1c,m2c).shape
print(torch.isclose(module.matmul(m1c,m2c), m1c@m2c).all())

t0 = time.perf_counter_ns()
for i in range(10):
    module.matmul(m1c,m2c)
torch.cuda.synchronize()
t1 = time.perf_counter_ns()
print((t1-t0) /10/1000, "µs") 


cuda_src = cuda_begin + r'''
__global__ void matmul_k(float *m, float *n, float *out, int h, int w, int k, int tw) {
    int tc=threadIdx.x, tr=threadIdx.y;
    int r=blockIdx.y*blockDim.y+tr, c=blockIdx.x*blockDim.x+tc;

    extern __shared__ float ms[];
    float *ns = &ms[tw*tw];

    float p = 0.0f;
    for (int ph = 0; ph < cdiv(k,tw); ++ph) {
        int idx = ph*tw;
        ms[tr*tw + tc] = r<h && idx+tc<k ? m[ tc+idx + r*k ] : 0.0f;
        ns[tr*tw + tc] = c<w && idx+tr<k ? n[(tr+idx)*w + c] : 0.0f;
        __syncthreads();
        for (int i=0; i<tw; ++i) p += ms[tr*tw + i] * ns[tw*i + tc];
       __syncthreads();
    }
    if (r<h && c<w) out[r*w + c] = p;
}
'''
        

cuda_src += r'''
torch::Tensor matmul_dyn(torch::Tensor m, torch::Tensor n) {
    CHECK_INPUT(m); CHECK_INPUT(n);
    int h=m.size(0), w=n.size(1), k=m.size(1);
    TORCH_CHECK(k==n.size(0), "Size mismatch!");
    auto output = torch::zeros({h, w}, m.options());

    /*
    // Commented out section demonstrating basic idea of dynamic size calculation
    cudaDeviceProp devProp;
    CUDA_ERR(cudaGetDeviceProperties(&devProp, 0));
    int maxThreads = devProp.maxThreadsPerBlock;
    size_t requiredSize = static_cast(maxThreads) * 2 * sizeof(float);
    size_t size = min(devProp.sharedMemPerBlock, requiredSize);
    int TW = std::sqrt(maxThreads);
    */

    // We just set size fixed for now
    int TW = 16;
    size_t size = TW*TW * 2 * sizeof(float);
    dim3 tpb(TW,TW);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    matmul_k<<>>(
        m.data_ptr(), n.data_ptr(), output.data_ptr(), h, w, k, TW);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
'''
fname = 'matmul_dyn'
#cpp_src = get_sig(fname, cuda_src)    
cpp_src = "torch::Tensor matmul_dyn(torch::Tensor m, torch::Tensor n);"
module = load_cuda(cuda_src, cpp_src, [fname], opt=True)
print(torch.isclose(module.matmul_dyn(m1c,m2c), m1c@m2c).all())
    
t0 = time.perf_counter_ns()
for i in range(10):
    module.matmul_dyn(m1c,m2c)
torch.cuda.synchronize()
t1 = time.perf_counter_ns()
print((t1-t0) /10/1000, "µs") 


