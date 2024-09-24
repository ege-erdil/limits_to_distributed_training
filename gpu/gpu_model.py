import numpy as np
from functools import lru_cache
from copy import deepcopy

def convert_to_np_array(variable, dtype):
    if not isinstance(variable, np.ndarray):
        return np.asarray([variable], dtype=dtype)
    else:
        return variable

ki = 1024
Mi = 1024*1024
tensor_core_chunk_size = 16 # TODO: should perhaps be GPU-specific

class GPU:
  def __init__(self, name, bitwidth, flop_per_clock_per_thread, register_bytes_per_processing_block, num_sms, l2_Bps, l2_bytes, global_Bps, effective_utilization, max_clock_Hz,
                distributed_shared_memory, memory_bytes, latency_per_matmul_seconds, network_bandwidths_per_level_Bps, network_latency_per_level_seconds, level_sizes):
      # Network bandwidths are bidirectional.

      self.name = name
      self.bitwidth = bitwidth
      self.flop_per_clock_per_thread = flop_per_clock_per_thread
      self.register_bytes_per_processing_block = register_bytes_per_processing_block
      self.num_sms = num_sms
      self.l2_Bps = l2_Bps
      self.l2_bytes = l2_bytes
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
      self.useful_l2_bytes = 0.75*self.l2_bytes
      self.shared_Bps = 32*4*self.num_sms*self.clock_Hz # 32 banks per SM, 4 bytes per cycle per bank

      self.num_threads = 128*self.num_sms
      self.flop_per_s = self.flop_per_clock_per_thread[self.bitwidth]*self.num_threads*self.clock_Hz
      self.max_flop_per_s = self.flop_per_clock_per_thread[self.bitwidth]*self.num_threads*self.max_clock_Hz

      self.dsmem = self.distributed_shared_memory

      self.latency_per_matmul_seconds = latency_per_matmul_seconds

  def __hash__(self):
      return hash(self.name)

  @lru_cache(maxsize=None)
  def matmul_time_seconds(self, d1, d2, b, for_weight_grads=False):
    """The time for a matrix multiplication either of:

      - Weights of shape d1xd2 by activations of shape d2xb, to compute
          Y = WX,

      - Weights transposed to shape d1xd2 by activation gradients of shape d2xb,
        to compute
          ∂L/∂X = W^T(∂L/∂Y),

       or

      - Activation gradients of shape d1xb by activations transposed to shape
        bxd2, to compute
          ∂L/∂W = (∂L/∂Y)X^T,
        in which case for_weight_grads is set to True.

    The microbatch size b will normally be the smallest dimension, since
    pipeline and data parallelism compete to slice this dimension finely. Thus
    we assume the kernel is tiled at all levels of the memory hierarchy
    according to the weight matrix dimensions d1xd2 (even if this is not the
    matmul output surface) to permit more parallelism across SMs.

    We also assume that the weights, along with their gradients, are permanently
    stored (except for data-parallel all-reduces, not modeled here) at the
    lowest level (among global, L2 cache, and registers) of the memory hierarchy
    in which they can fit, with no data movement required higher up."""
    # Data movement between global memory and L2 cache.
    global_tofrom_l2_io_bytes, _ = self._matmul_io_bytes_between_levels(
        d1, d2, b,
        self.useful_l2_bytes,
        for_weight_grads=for_weight_grads)

    # Data movement between L2 cache and L1 cache/shared memory (weight matrix
    # tiles are assumed stored in registers for computation; L1 cache/shared
    # memory is just used as a transient buffer). If distributed shared memory
    # is present, this reduces the burden on L2 for redundant loads or stores by
    # multiple SMs.
    l2_tofrom_sm_io_bytes, dsmem_io_bytes = self._matmul_io_bytes_between_levels(
        d1, d2, b,
        4*self.useful_register_bytes_per_processing_block,
        num_concurrent=self.num_sms,
        distributed_memory=self.dsmem,
        for_weight_grads=for_weight_grads)

    # Data movement between L1 cache/shared memory and registers.
    sm_tofrom_registers_io_bytes, _ = self._matmul_io_bytes_between_levels(
        d1, d2, b,
        self.useful_register_bytes_per_processing_block,
        num_concurrent=4*self.num_sms,
        for_weight_grads=for_weight_grads)

    global_io_bytes = global_tofrom_l2_io_bytes
    l2_io_bytes = global_tofrom_l2_io_bytes + l2_tofrom_sm_io_bytes
    sm_io_bytes = l2_tofrom_sm_io_bytes + dsmem_io_bytes + sm_tofrom_registers_io_bytes

    # Pad the dimensions to the tensor core shape before computing effective
    # FLOP.
    d1_chunks = np.ceil(d1/tensor_core_chunk_size)
    d2_chunks = np.ceil(d2/tensor_core_chunk_size)
    b_chunks = np.ceil(b/tensor_core_chunk_size)
    flop = 2*d1_chunks*d2_chunks*b_chunks*tensor_core_chunk_size**3

    # Calculate durations as the maximum of all data movement and arithmetic
    # times, plus a fixed latency.
    time_s = np.maximum(np.maximum(global_io_bytes/self.global_Bps, l2_io_bytes/self.l2_Bps),
                        np.maximum(sm_io_bytes/self.shared_Bps, flop/self.flop_per_s))
    time_s += self.latency_per_matmul_seconds
    return time_s

  def _tile(self, m, n, max_tile_bytes):
      """Ignoring most discrete constraints (to avoid combinatorial search),
      find the minimum total tile count such that the tiles fit in the lower
      level, and then the closest to a square we can make the tile shape. Then
      return the resulting number of tile rows and columns. This is used to
      determine the number of redundant accesses to the higher level for each
      activation (or activation gradient) value."""
      words = m*n
      bytes = words*self.bytewidth
      tile_count = np.ceil(bytes/max_tile_bytes)
      words_per_tile = words/tile_count
      tile_m = np.maximum(np.minimum(m, np.sqrt(words_per_tile)),
                          words_per_tile/n)
      tile_n = words_per_tile/tile_m
      return m/tile_m, n/tile_n

  def _matmul_io_bytes_between_levels(
          self, d1, d2, b,
          max_tile_bytes, num_concurrent=1,
          distributed_memory=False,
          for_weight_grads=False):
    """The IO volume between adjacent levels of the memory hierarchy for a
    matrix multiplication with shapes specified by d1, d2, b, and
    for_weight_grads as in matmul_time_seconds. There are num_concurrent
    instances (e.g. SMs or processing blocks) of tiles being processed, each
    handling a weight matrix tile of at most max_tile_bytes.

    If distributed_memory is True, this acts like Hopper's distributed shared
    memory, allowing redundant accesses to the higher level to be avoided (e.g.
    avoiding redundant loads from L2 cache to different SMs, since the SMs can
    communicate amongst each other). In this case, a second return value is
    included for the IO volume on the distributed network. Otherwise the second
    return value is 0."""
    weight_words = d1*d2
    weight_bytes = weight_words*self.bytewidth
    lefthand_activation_bytes = d1*b*self.bytewidth
    righthand_activation_bytes = d2*b*self.bytewidth

    weight_grad_bytes = weight_bytes
    weights_and_grads_fit_in_lower_level = \
        weight_bytes + weight_grad_bytes <= max_tile_bytes*num_concurrent

    # Determine the effective tiling shape for the inter-level data movement and
    # the within-level data movement for the distributed network.
    if distributed_memory:
        num_big_rows, num_big_cols = self._tile(d1, d2, max_tile_bytes*num_concurrent)
        num_small_rows, num_small_cols = self._tile(d1, d2, max_tile_bytes)
    else:
        num_big_rows, num_big_cols = self._tile(d1, d2, max_tile_bytes)
        num_small_rows, num_small_cols = 1, 1

    # Now figure out the required total IO volume both between the hierarchy
    # levels, as well as on the distributed network at the lower level if it
    # exists.
    interlevel_io_bytes = 0
    dist_io_bytes = 0
    if not weights_and_grads_fit_in_lower_level:
        if for_weight_grads:
            # The previously accumulated weight gradients will have to be loaded
            # (in tile-sized chunks) from the higher to lower level, and then
            # the newly accumulated weight gradients will have to be stored back
            # (also in tile-sized chunks) to the higher level.
            interlevel_io_bytes += 2*weight_grad_bytes
        else:
            # The weights will have to be loaded (in tile-sized chunks) from
            # the higher to lower level.
            interlevel_io_bytes += weight_bytes

    # Each element of the righthand side activations X (or X^T, or activation
    # gradients ∂L/∂Y) of size d2xb will have to be loaded from the higher to
    # lower level once for each big row of weight tiles, and transferred on the
    # distributed memory network for each small row of weight tiles beyond the
    # first.
    interlevel_io_bytes += num_big_rows*righthand_activation_bytes
    dist_io_bytes += (num_small_rows - 1)*righthand_activation_bytes

    if for_weight_grads:
        # Each element of the lefthand side activation gradients ∂L/∂Y of size
        # d1xb will have to be loaded from the higher to lower level once for
        # each column of weight tiles, and transferred on the distributed memory
        # network for each column of weight tiles beyond the first.
        interlevel_io_bytes += num_big_cols*lefthand_activation_bytes
        dist_io_bytes += (num_small_cols - 1)*lefthand_activation_bytes
    else:
        # Each element of the lefthand side output activations Y (or output
        # activation gradients ∂L/∂X) of size d1xb will have to be accumulated
        # once for each big column of weight tiles, and reduce-scattered on the
        # distributed memory network across small columns of weight tiles. The
        # former involves a load from the higher to lower level to read the
        # previously accumulated value (except the very first time), then a
        # store from the lower to higher level to write the newly accumulated
        # value. The latter involves a single transfer on the distributed memory
        # network for each small column beyond the first.
        interlevel_io_bytes += (2*num_big_cols - 1)*lefthand_activation_bytes
        dist_io_bytes += (num_small_cols - 1)*lefthand_activation_bytes

    # Effective data movement on the distributed network has to be doubled
    # to count both sides.
    return interlevel_io_bytes, 2*dist_io_bytes

V100_SXM2 = GPU(
      name = "V100 SXM",
      bitwidth = 16,
      flop_per_clock_per_thread = {32: 1, 16: 8},
      register_bytes_per_processing_block = 64*ki,
      num_sms = 80,
      l2_Bps = 2155e9,
      l2_bytes = 6*Mi,
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
      l2_bytes = 40*Mi,
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
      l2_bytes = 50*Mi,
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
      l2_Bps = 6441e9,  # Reusing PCIe number scaled by number of SMs, not sure it's right but doesn't matter much due to DSMEM.
      l2_bytes = 50*Mi,
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