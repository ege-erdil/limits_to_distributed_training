import numpy as np
from functools import lru_cache
from copy import deepcopy

def convert_to_np_array(variable, dtype):
    if not isinstance(variable, np.ndarray):
        return np.asarray([variable], dtype=dtype)
    else:
        return variable

ki = 1024

class GPU:
  def __init__(self, name, bitwidth, flop_per_clock_per_thread, register_bytes_per_processing_block, num_sms, l2_Bps, global_Bps, effective_utilization, max_clock_Hz,
                distributed_shared_memory, memory_bytes, latency_per_matmul_seconds, network_bandwidths_per_level_Bps, network_latency_per_level_seconds, level_sizes):
      # Network bandwidths are bidirectional.

      self.name = name
      self.bitwidth = bitwidth
      self.flop_per_clock_per_thread = flop_per_clock_per_thread
      self.register_bytes_per_processing_block = register_bytes_per_processing_block
      self.num_sms = num_sms
      self.l2_Bps = l2_Bps
      self.global_Bps = global_Bps
      self.max_clock_Hz = max_clock_Hz
      self.clock_Hz = max_clock_Hz*effective_utilization
      self.distributed_shared_memory = distributed_shared_memory
      self.memory_bytes = memory_bytes
      self.network_bandwidths_per_level_Bps = network_bandwidths_per_level_Bps
      self.network_latency_per_level_seconds = network_latency_per_level_seconds
      self.level_sizes = level_sizes

      self.bytewidth = self.bitwidth/8
      self.useful_register_bytes_per_processing_block = 0.75*self.register_bytes_per_processing_block
      self.shared_Bps = 32*4*self.num_sms*self.clock_Hz # 32 banks per SM, 4 bytes per cycle per bank

      self.num_threads = 128*self.num_sms
      self.flop_per_s = self.flop_per_clock_per_thread[self.bitwidth]*self.num_threads*self.clock_Hz
      self.max_flop_per_s = self.flop_per_clock_per_thread[self.bitwidth]*self.num_threads*self.max_clock_Hz

      self.dsmem = self.distributed_shared_memory

      self.latency_per_matmul_seconds = latency_per_matmul_seconds

  def __hash__(self):
      return hash(self.name)

  def shape(self, M, N, warp_reg_bytes):
      # We always assume all SMs are being used. One might think for small enough matrices
      # one would instead use fewer SMs to get more L2 bandwidth per SM, but total L2
      # bandwidth seems to be close to linear in the number of SMs, so this doesn't help.
      #
      # Source: my own experiments on an A10 (l2_bandwidth.cu) as well as
      # https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
      assert np.all(M >= N)
      num_warps = 4*self.num_sms

      warp_area = np.minimum(M*N/num_warps, warp_reg_bytes/self.bytewidth)
      sm_area = 4*warp_area
      combined_area = self.num_sms*sm_area

      # These are optimistic shapes, ignoring discrete constraints
      # and address alignment.
      N_warp = np.minimum(N, np.sqrt(warp_area.astype(float)))
      M_warp = warp_area/N_warp

      N_sm = np.minimum(N, np.sqrt(sm_area.astype(float)))
      M_sm = sm_area/N_sm

      N_combined = np.minimum(N, np.sqrt(combined_area.astype(float)))
      M_combined = combined_area/N_combined

      return M_combined, N_combined, M_sm, N_sm, M_warp, N_warp

  def io_intensity(self, M, N, K):
      return self.bytewidth/2 * (1/M + 1/N + 1/K)

  def io_intensities(self, M, N, M_combined, N_combined, M_sm, N_sm, M_warp, N_warp, K):
      # Load M × K and K × N from global memory, then store the result
      # tile of size M × N.
      # TODO: Currently assuming an infinite L2 cache. In reality we should
      #       determine M_L2 and N_L2 and use those instead of M and N.
      ioint_global = self.io_intensity(M, N, K)

      if self.dsmem:
          # In this case, once a single SM has the data, we never have to hit
          # L2 again. So L2 bandwidth = global bandwidth.
          #
          # TODO: Not clear if DSMEM supports a multicast mechanism. If so, then
          #       transmission is ~free. If not, then it's best to think of DSMEM
          #       + L2 as a combined pool of bandwidth for communication between
          #       SMs once it's already loaded from global memory. We currently
          #       assume multicast.
          #
          # p. 10 of https://docs.nvidia.com/cuda/pdf/Hopper_Tuning_Guide.pdf:
          # "Distributed Shared Memory can be used by an SM simultaneously with L2
          # cache accesses. This can benefit applications that need to communicate
          # data between SMs by utilizing the combined bandwidth of both
          # distributed shared memory and L2."
          #
          # This implies the DSMEM network doesn't support full bisection bandwidth,
          # however it's not clear if this means it doesn't support multicast.
          ioint_l2 = self.io_intensity(M_combined, N_combined, K)
      else:
          # In this case, separate hits to L2 are required for each SM.
          ioint_l2 = self.io_intensity(M_sm, N_sm, K)

      # Store part of A and B from global into shared, then load from shared for each warp.
      # We treat K as infinite because it's not necessary to write the result back to shared.
      ioint_shared = self.io_intensity(M_sm, N_sm, np.inf) + self.io_intensity(M_warp, N_warp, np.inf)

      return np.broadcast_arrays(ioint_global, ioint_l2, ioint_shared)

  def io_intensities_gpu(self, M, N, K, warp_reg_bytes):
      M_combined, N_combined, M_sm, N_sm, M_warp, N_warp = self.shape(M, N, warp_reg_bytes)
      return self.io_intensities(M, N, M_combined, N_combined, M_sm, N_sm, M_warp, N_warp, K)

  def get_flop_throughput(self, M, N, K, warp_reg_bytes):
      M, N, K = np.broadcast_arrays(M, N, K)

      io_intensity_global, io_intensity_l2, io_intensity_shared =\
            self.io_intensities_gpu(M, N, K, warp_reg_bytes)

      return np.minimum(np.minimum(self.flop_per_s, self.global_Bps/io_intensity_global),
                        np.minimum(self.l2_Bps/io_intensity_l2, self.shared_Bps/io_intensity_shared))

  @lru_cache(maxsize=None)
  def matmul_time_seconds(self, m, k, n):
    if not (m >= n and n >= k): # if not m >= n >= k, reorder the inputs and call the function again; we don't stay in the same function scope so that lru_cache can cache the value for all permutations
      min_dim, mid_dim, max_dim = sorted([m, k, n])
      return self.matmul_time_seconds(max_dim, min_dim, mid_dim)
    else:
      return self.latency_per_matmul_seconds + 2*m*k*n/self.get_flop_throughput(convert_to_np_array(m, dtype=np.int64), convert_to_np_array(n, dtype=np.int64), convert_to_np_array(k, dtype=np.int64), self.useful_register_bytes_per_processing_block)[0]

V100_SXM2 = GPU(
      name = "V100 SXM",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 1, 16: 8},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 80,
      l2_Bps = 2155e9,
      global_Bps = 900e9,
      max_clock_Hz = 1533e6,
      effective_utilization = 0.9, # empirically observed hardware utilization rate when running a long sequence of big matmuls
      distributed_shared_memory = False,
      memory_bytes = 3.2e10,
      latency_per_matmul_seconds = 4.5e-6,
      # 300 GB/s bidirectional from the V100 NVLink bandwidth, 6.25 GB/s = 50 Gb/s unidirectional (12.5 GB/s bidirectional) from the 4x100Gb/s EDR IB cards on a DGX-1 system
      # Source: https://images.nvidia.com/content/pdf/dgx1-v100-system-architecture-whitepaper.pdf
      network_bandwidths_per_level_Bps = [3e11, 1.25e10], 
      network_latency_per_level_seconds = [5e-6, 5e-6],
      level_sizes = (8,)
)
A100 = GPU(
      name = "A100 SXM",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 1, 16: 16, 8: 32},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 108,
      # l2_Bps = 7050e9 # partitioned on Ampere, but we don't model this
      # profiled by https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
      l2_Bps = 5603e9,
      global_Bps = 1555e9,
      max_clock_Hz = 1410e6,
      effective_utilization = 0.85, # empirically observed hardware utilization rate when running a long sequence of big matmuls
      distributed_shared_memory = False,
      memory_bytes = 4e10,
      latency_per_matmul_seconds = 4.5e-6,
      # 600 GB/s bidirectional from the A100 NVLink bandwidth, 25 GB/s = 200 Gb/s unidirectional (50 GB/s bidirectional) per GPU from the 8x200Gb/s ConnectX-6 cards on a DGX A100 system
      # Source: https://download.boston.co.uk/downloads/3/8/6/386750a7-52cd-4872-95e4-7196ab92b51c/DGX%20A100%20System%20Architecture%20Whitepaper.pdf (seems to be removed from Nvidia's web site)
      network_bandwidths_per_level_Bps = [6e11, 5e10],
      network_latency_per_level_seconds = [5e-6, 5e-6],
      level_sizes = (8,)
)
H100_PCIe = GPU(
      name = "H100 PCIe",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 32, 8: 64},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 114,
      l2_Bps = 5563e9,
      global_Bps = 2039e9, # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1620e6,
      effective_utilization = 0.55, # empirically observed hardware utilization rate when running a long sequence of big matmuls
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 4.5e-6,
      # There's no standard H100 PCIe system like there is a DGX H100, but we assume the same interconnects as the DGX H100 below.
      # Source: https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf
      network_bandwidths_per_level_Bps = [9e11, 1e11],
      network_latency_per_level_seconds = [5e-6, 5e-6],
      level_sizes = (8,)
)

H100_SXM5 = GPU(
      name = "H100 SXM",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 32, 8: 64},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 132,
      l2_Bps = 6441e9,  # Reusing PCIe number scaled by number of SMs, not sure it's right but doesn't matter much.
      global_Bps = 3352e9, # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1830e6,
      effective_utilization = 0.7, # empirically observed hardware utilization rate when running a long sequence of big matmuls
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 4.5e-6,
      # 900 GB/s from the H100 NVLink bandwidth, 50 GB/s = 400 Gb/s unidirectional (100 GB/s bidirectional) per GPU from the 8x400Gb/s ConnectX-7 cards on a DGX H100 system
      # Source: https://nvdam.widen.net/s/95bdhpsgrs/nvidia_h100_tensor_core_gpu_architecture_whitepaper_v1.03
      network_bandwidths_per_level_Bps = [9e11, 1e11],
      network_latency_per_level_seconds = [5e-6, 5e-6],
      level_sizes = (8,)
)

# Some variations on the H100 SXM5 with different hypothetical network configurations.

H100_SXM5_Superpod = deepcopy(H100_SXM5)
H100_SXM5_Superpod.name = "H100 SXM Superpod"
H100_SXM5_Superpod.level_sizes = (256,)

H100_SXM5_Zero_Latency = deepcopy(H100_SXM5)
H100_SXM5_Zero_Latency.name = "H100 SXM Zero Latency"
H100_SXM5_Zero_Latency.latency_per_matmul_seconds = 0
H100_SXM5_Zero_Latency.network_latency_per_level_seconds = [0, 0]

H100_SXM5_Superpod_Zero_Latency = deepcopy(H100_SXM5_Zero_Latency)
H100_SXM5_Superpod_Zero_Latency.name = "H100 SXM Superpod ZL"
H100_SXM5_Superpod_Zero_Latency.level_sizes = (256,)

H100_SXM5_Global_NVLink = deepcopy(H100_SXM5)
H100_SXM5_Global_NVLink.name = "H100 SXM Global NVLink"
H100_SXM5_Global_NVLink.network_bandwidths_per_level_Bps = [9e11]
H100_SXM5_Global_NVLink.network_latency_per_level_seconds = [5e-6]
H100_SXM5_Global_NVLink.level_sizes = ()

H100_SXM5_Global_NVLink_Zero_Latency = deepcopy(H100_SXM5)
H100_SXM5_Global_NVLink_Zero_Latency.name = "H100 SXM Global NVLink and ZL"
H100_SXM5_Global_NVLink_Zero_Latency.latency_per_matmul_seconds = 0
H100_SXM5_Global_NVLink_Zero_Latency.network_bandwidths_per_level_Bps = [9e11]
H100_SXM5_Global_NVLink_Zero_Latency.network_latency_per_level_seconds = [0]
H100_SXM5_Global_NVLink_Zero_Latency.level_sizes = ()

H100_SXM5_Infinite_Network_Zero_Latency = deepcopy(H100_SXM5_Global_NVLink_Zero_Latency)
H100_SXM5_Infinite_Network_Zero_Latency.name = "H100 SXM Infinite Network and ZL"
H100_SXM5_Infinite_Network_Zero_Latency.network_bandwidths_per_level_Bps = [np.inf]

H100_SXM5_Superpod_Singleton = deepcopy(H100_SXM5)
H100_SXM5_Superpod_Singleton.name = "H100 SXM Superpod Singleton"
H100_SXM5_Superpod_Singleton.num_sms *= 256
H100_SXM5_Superpod_Singleton.l2_Bps *= 256
H100_SXM5_Superpod_Singleton.global_Bps *= 256
H100_SXM5_Superpod_Singleton.memory_bytes *= 256
H100_SXM5_Superpod_Singleton.network_bandwidths_per_level_Bps = [5e10*256]
H100_SXM5_Superpod_Singleton.network_latency_per_level_seconds = [5e-6]
H100_SXM5_Superpod_Singleton.level_sizes = ()

gpu_list = [V100_SXM2, A100, H100_SXM5, H100_SXM5_Superpod, H100_SXM5_Superpod_Singleton]
gpu_dict = {gpu.name: gpu for gpu in gpu_list}