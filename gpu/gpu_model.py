import numpy as np
from functools import lru_cache

def convert_to_np_array(variable, dtype):
    if not isinstance(variable, np.ndarray):
        return np.asarray([variable], dtype=dtype)
    else:
        return variable

ki = 1024

class GPU:
  def __init__(self, name, bitwidth, flop_per_clock_per_thread, register_bytes_per_processing_block, num_sms, l2_Bps, global_Bps, effective_utilization, max_clock_Hz,
                distributed_shared_memory, memory_bytes, latency_per_matmul_seconds, network_bandwidths_per_level_Bps, network_latency_per_level_seconds, level_sizes):

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
      n = len(M)

      io_intensity_global, io_intensity_l2, io_intensity_shared =\
      self.io_intensities_gpu(M, N, K, warp_reg_bytes)

      flop_throughput_bigmk_global = np.minimum(self.flop_per_s, self.global_Bps/io_intensity_global)
      flop_throughput_bigmk_l2 = np.minimum(self.flop_per_s, self.l2_Bps/io_intensity_l2)
      flop_throughput_bigmk_shared = np.minimum(self.flop_per_s, self.shared_Bps/io_intensity_shared)
      flop_throughput_bigmk = np.minimum(np.minimum(flop_throughput_bigmk_global, flop_throughput_bigmk_l2),
                                        flop_throughput_bigmk_shared)

      return flop_throughput_bigmk

  def matmul_arithmetic_time_seconds(self, m, k, n): # returns how much time a matrix multiplication should take if we achieve perfect FLOP utilization
    return 2*m*k*n/(self.max_flop_per_s)

  @lru_cache(maxsize=None)
  def matmul_time_seconds(self, m, k, n):
    if not (m >= n and n >= k): # if not m >= n >= k, reorder the inputs and call the function again; we don't stay in the same function scope so that lru_cache can cache the value for all permutations
      min_dim, mid_dim, max_dim = sorted([m, k, n])
      return self.matmul_time_seconds(max_dim, min_dim, mid_dim)
    else:
      return self.latency_per_matmul_seconds + 2*m*k*n/self.get_flop_throughput(convert_to_np_array(m, dtype=np.int64), convert_to_np_array(n, dtype=np.int64), convert_to_np_array(k, dtype=np.int64), self.useful_register_bytes_per_processing_block)[0]

  def matmul_time_seconds_vectorized(self, m, k, n):
    # print(np.shape(m), np.shape(k), np.shape(n))
    dimensions = np.stack((m, k, n))
    max_val = np.max(dimensions, axis=1)
    min_val = np.min(dimensions, axis=1)
    mid_val = np.median(dimensions, axis=1)

    return 2*m*k*n/self.get_flop_throughput(max_val, mid_val, min_val, self.useful_register_bytes_per_processing_block)[0]

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
      network_bandwidths_per_level_Bps = [3e11, 2.5e10], # 300 GB/s from the V100 NVLink bandwidth, 12.5 GB/s from extrapolating the 2x difference between H100 and A100 one generation backwards
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
      network_bandwidths_per_level_Bps = [6e11, 5e10], # 600 GB/s from the A100 NVLink bandwidth, 25 GB/s = 200 Gb/s per GPU from the 8x200Gb/s ConnectX-7 cards on a DGX A100 system
      network_latency_per_level_seconds = [5e-6, 5e-6],
      level_sizes = (8,)
)
H100_PCIe = GPU(
      name = "H100 PCIe",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 29.54, 8: 59.01},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 114,
      l2_Bps = 5563e9,
      global_Bps = 2039e9,
                        # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1755e6,
      effective_utilization = 0.55, # empirically observed hardware utilization rate when running a long sequence of big matmuls
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 4.5e-6,
      network_bandwidths_per_level_Bps = [9e11, 1e11], # 900 GB/s from the H100 NVLink bandwidth, 50 GB/s = 400 Gb/s per GPU from the 8x400Gb/s ConnectX-7 cards on a DGX H100 system
      network_latency_per_level_seconds = [5e-6, 5e-6],
      level_sizes = (8,)
)
H100_SXM5 = GPU(
      name = "H100 SXM",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 29.58, 8: 59.16},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 132,
      l2_Bps = 6441e9,  # Reusing PCIe number scaled by number of SMs, not sure it's right but doesn't matter much.
      global_Bps = 3352e9,  # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1980e6,
      effective_utilization = 0.7, # empirically observed hardware utilization rate when running a long sequence of big matmuls
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 4.5e-6,
      network_bandwidths_per_level_Bps = [9e11, 1e11], # 900 GB/s from the H100 NVLink bandwidth, 50 GB/s = 400 Gb/s per GPU from the 8x400Gb/s ConnectX-7 VPI cards on a DGX H100 system
      network_latency_per_level_seconds = [5e-6, 5e-6],
      level_sizes = (8,)
)
H100_SXM5_Superpod = GPU(
      name = "H100 SXM Superpod",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 29.58, 8: 59.16},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 132,
      l2_Bps = 6441e9,  # profiled by https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
                      # A little skeptical of this number as I can achieve at least ~5.7 TB/s on an A10, and that's
                      # probably not the best case.
                      #
                      # Doesn't matter much since we're generally not L2-bound.
      global_Bps = 3352e9,  # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1980e6,
      effective_utilization = 0.7,
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 4.5e-6,
      network_bandwidths_per_level_Bps = [9e11, 1e11],
      network_latency_per_level_seconds = [5e-6, 5e-6],
      level_sizes = (256,)
)
H100_SXM5_Zero_Latency = GPU(
      name = "H100 SXM Zero Latency",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 29.58, 8: 59.16},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 132,
      l2_Bps = 6441e9,  # Reusing PCIe number scaled by number of SMs, not sure it's right but doesn't matter much.
      global_Bps = 3352e9,  # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1980e6,
      effective_utilization = 0.7, # empirically observed hardware utilization rate when running a long sequence of big matmuls
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 0,
      network_bandwidths_per_level_Bps = [9e11, 1e11], # 900 GB/s from the H100 NVLink bandwidth, 50 GB/s = 400 Gb/s per GPU from the 8x400Gb/s ConnectX-7 VPI cards on a DGX H100 system
      network_latency_per_level_seconds = [0, 0],
      level_sizes = (8,)
)
H100_SXM5_Superpod_Zero_Latency = GPU(
      name = "H100 SXM Superpod ZL",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 29.58, 8: 59.16},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 132,
      l2_Bps = 6441e9,  # profiled by https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
                      # A little skeptical of this number as I can achieve at least ~5.7 TB/s on an A10, and that's
                      # probably not the best case.
                      #
                      # Doesn't matter much since we're generally not L2-bound.
      global_Bps = 3352e9,  # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1980e6,
      effective_utilization = 0.7,
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 0,
      network_bandwidths_per_level_Bps = [9e11, 1e11],
      network_latency_per_level_seconds = [0, 0],
      level_sizes = (256,)
)
H100_SXM5_Global_NVLink = GPU(
      name = "H100 SXM Global NVLink",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 29.58, 8: 59.16},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 132,
      l2_Bps = 6441e9,  # profiled by https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
                      # A little skeptical of this number as I can achieve at least ~5.7 TB/s on an A10, and that's
                      # probably not the best case.
                      #
                      # Doesn't matter much since we're generally not L2-bound.
      global_Bps = 3352e9,  # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1980e6,
      effective_utilization = 0.7,
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 4.5e-6,
      network_bandwidths_per_level_Bps = [9e11],
      network_latency_per_level_seconds = [5e-6],
      level_sizes = ()
)
H100_SXM5_Global_NVLink_Zero_Latency = GPU(
      name = "H100 SXM Global NVLink and ZL",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 29.58, 8: 59.16},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 132,
      l2_Bps = 6441e9,  # profiled by https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
                      # A little skeptical of this number as I can achieve at least ~5.7 TB/s on an A10, and that's
                      # probably not the best case.
                      #
                      # Doesn't matter much since we're generally not L2-bound.
      global_Bps = 3352e9,  # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1980e6,
      effective_utilization = 0.7,
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 0,
      network_bandwidths_per_level_Bps = [9e11],
      network_latency_per_level_seconds = [0],
      level_sizes = ()
)
H100_SXM5_Infinite_Network_Zero_Latency = GPU(
      name = "H100 SXM Infinite Network and ZL",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 29.58, 8: 59.16},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 132,
      l2_Bps = 6441e9,  # profiled by https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
                      # A little skeptical of this number as I can achieve at least ~5.7 TB/s on an A10, and that's
                      # probably not the best case.
                      #
                      # Doesn't matter much since we're generally not L2-bound.
      global_Bps = 3352e9,  # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1980e6,
      effective_utilization = 0.7,
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 0,
      network_bandwidths_per_level_Bps = [np.inf],
      network_latency_per_level_seconds = [0],
      level_sizes = ()
)
NVIDIA_2028_Superpod = GPU(
      name = "2028 NVIDIA GPU Guess Superpod",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 1.5*2, 16: 1.5*29.58, 8: 1.5*59.16},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 2*132,
      l2_Bps = 2*6441e9,
      global_Bps = 2*3352e9,
      max_clock_Hz = 1.33*1980e6,
      effective_utilization = 0.7,
      distributed_shared_memory = True,
      memory_bytes = 4*8e10,
      latency_per_matmul_seconds = 4.5e-6,
      network_bandwidths_per_level_Bps = [2*9e11, 2*4.5e11, 2*5e10],
      network_latency_per_level_seconds = [5e-6, 5e-6, 5e-6],
      level_sizes = (8, 32)
)
H100_SXM5_Superpod_Singleton = GPU(
      name = "H100 SXM Superpod Singleton",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 29.58, 8: 59.16},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 132*256,
      l2_Bps = 6441e9*256,  # profiled by https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/
                      # A little skeptical of this number as I can achieve at least ~5.7 TB/s on an A10, and that's
                      # probably not the best case.
                      #
                      # Doesn't matter much since we're generally not L2-bound.
      global_Bps = 3352e9*256,  # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1980e6,
      effective_utilization = 0.7,
      distributed_shared_memory = True,
      memory_bytes = 8e10*256,
      latency_per_matmul_seconds = 4.5e-6,
      network_bandwidths_per_level_Bps = [5e10*256],
      network_latency_per_level_seconds = [5e-6, 5e-6, 5e-6],
      level_sizes = ()
)
H100_Datacenter = GPU(
      name = "H100 SXM Datacenter",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 2, 16: 29.58, 8: 59.16},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 132,
      l2_Bps = 6441e9,  # Reusing PCIe number scaled by number of SMs, not sure it's right but doesn't matter much.
      global_Bps = 3352e9,  # https://resources.nvidia.com/en-us-tensor-core
      max_clock_Hz = 1980e6,
      effective_utilization = 0.7, # empirically observed hardware utilization rate when running a long sequence of big matmuls
      distributed_shared_memory = True,
      memory_bytes = 8e10,
      latency_per_matmul_seconds = 4.5e-6,
      network_bandwidths_per_level_Bps = [9e11, 1e11, 1e9], # 900 GB/s from the H100 NVLink bandwidth, 50 GB/s = 400 Gb/s per GPU from the 8x400Gb/s ConnectX-7 VPI cards on a DGX H100 system
                                                            # we assume 1 GB/s per GPU and 1 ms latency between datacenters of 32K GPUs, which is likely too pessimistic
                                                            # still, it serves as a useful check on what kind of training run is feasible to achieve even with very poor connections between datacenters
      network_latency_per_level_seconds = [5e-6, 5e-6, 1e-3],
      level_sizes = (8, 4096)
)
gpu_list = [V100_SXM2, A100, H100_SXM5, H100_SXM5_Superpod, H100_SXM5_Superpod_Singleton, NVIDIA_2028_Superpod]
gpu_dict = {gpu.name: gpu for gpu in gpu_list}