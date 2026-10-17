#include <cuda_fp16.h>

typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef short int16_t;
typedef int int32_t;

template <typename T>
__device__ inline void corner_turn_body(
    const T *source, T *destination, long long nsamples, int nchans,
    long long destination_pitch, long long time_offset
) {
    __shared__ T tile[32][33];
    int channel_tiles = (nchans + 31) / 32;
    int channel_base = (blockIdx.x % channel_tiles) * 32;
    long long time_base = (blockIdx.x / channel_tiles) * 32LL;
    int channel = channel_base + threadIdx.x;
    long long time = time_base + threadIdx.y;
    for (int row = 0; row < 32; row += 8) {
        if (channel < nchans && time + row < nsamples) {
            tile[threadIdx.y + row][threadIdx.x] =
                source[(time + row) * nchans + channel];
        }
    }
    __syncthreads();
    channel = channel_base + threadIdx.y;
    time = time_base + threadIdx.x;
    for (int row = 0; row < 32; row += 8) {
        if (channel + row < nchans && time < nsamples) {
            destination[(long long)(channel + row) * destination_pitch
                        + time_offset + time] = tile[threadIdx.x][threadIdx.y + row];
        }
    }
}

#define DECLARE_CORNER_TURN(SIZE, TYPE) \
extern "C" __global__ void corner_turn_b##SIZE( \
    const TYPE *source, TYPE *destination, long long nsamples, int nchans, \
    long long destination_pitch, long long time_offset \
) { \
    corner_turn_body(source, destination, nsamples, nchans, destination_pitch, time_offset); \
}

DECLARE_CORNER_TURN(1, unsigned char)
DECLARE_CORNER_TURN(2, unsigned short)
DECLARE_CORNER_TURN(4, unsigned int)
DECLARE_CORNER_TURN(8, unsigned long long)

static const int TIME_TILE = 128;
static const int DM_TILE = 8;

template <typename T>
__device__ inline float sample_at(
    const T *source,
    const float *medians,
    const unsigned char *enabled,
    int local_channel,
    long long sample,
    long long source_start,
    long long source_count,
    long long valid_start,
    long long valid_stop,
    long long nspectra,
    double input_scale
) {
    if (!enabled[local_channel]) return 0.0f;
    if (
        sample >= 0 && sample < nspectra &&
        sample >= valid_start && sample < valid_stop &&
        sample >= source_start && sample - source_start < source_count
    ) {
        double value = static_cast<double>(
            source[(long long)local_channel * source_count + sample - source_start]
        );
        if (isfinite(value)) {
            float scaled = static_cast<float>(value / input_scale);
            if (isfinite(scaled)) return scaled;
        }
    }
    return medians[local_channel];
}

template <typename T>
__device__ inline void dedisp_ft_body(
    const T *source,
    const float *medians,
    const unsigned char *enabled,
    const long long *ft_delays,
    float *output,
    int nchans,
    int channel_start,
    int group_channels,
    long long source_start,
    long long source_count,
    long long valid_start,
    long long valid_stop,
    long long nspectra,
    long long output_start,
    int native_count,
    double input_scale
) {
    int time = blockIdx.x * 32 + threadIdx.x;
    int channel = blockIdx.y * 8 + threadIdx.y;
    if (time >= native_count || channel >= group_channels) return;
    int global_channel = channel_start + channel;
    long long sample = output_start + time + ft_delays[global_channel];
    output[(long long)time * nchans + global_channel] = sample_at(
        source, medians, enabled, channel, sample,
        source_start, source_count, valid_start, valid_stop, nspectra, input_scale
    );
}

template <typename T>
__device__ inline void dmt_tiled_body(
    const T *source,
    const float *medians,
    const unsigned char *enabled,
    const long long *delays,
    float *output,
    int nchans,
    int channel_start,
    int group_channels,
    int dmsteps,
    long long source_start,
    long long source_count,
    long long valid_start,
    long long valid_stop,
    long long nspectra,
    long long output_start,
    int native_count,
    int shared_capacity,
    int first_group,
    double input_scale,
    float *tile
) {
    int lane = threadIdx.x;
    int time_base = blockIdx.x * TIME_TILE;
    int time = time_base + lane;
    int dm_base = blockIdx.y * DM_TILE;
    int dm_count = min(DM_TILE, dmsteps - dm_base);
    float sums[DM_TILE] = {0.0f};

    int tile_times = min(TIME_TILE, native_count - time_base);
    bool use_shared = true;
    for (int local_channel = 0; local_channel < group_channels; ++local_channel) {
        int global_channel = channel_start + local_channel;
        long long minimum = delays[(long long)dm_base * nchans + global_channel];
        long long maximum = minimum;
        for (int d = 1; d < dm_count; ++d) {
            long long delay = delays[(long long)(dm_base + d) * nchans + global_channel];
            minimum = min(minimum, delay);
            maximum = max(maximum, delay);
        }
        if (maximum - minimum + tile_times > shared_capacity) {
            use_shared = false;
            break;
        }
    }

    if (use_shared) {
        for (int local_channel = 0; local_channel < group_channels; ++local_channel) {
            int global_channel = channel_start + local_channel;
            long long minimum = delays[(long long)dm_base * nchans + global_channel];
            long long maximum = minimum;
            for (int d = 1; d < dm_count; ++d) {
                long long delay = delays[(long long)(dm_base + d) * nchans + global_channel];
                minimum = min(minimum, delay);
                maximum = max(maximum, delay);
            }
            int count = tile_times + (int)(maximum - minimum);
            long long slab_start = output_start + time_base + minimum;
            for (int i = lane; i < count; i += TIME_TILE) {
                tile[i] = sample_at(
                    source, medians, enabled, local_channel, slab_start + i,
                    source_start, source_count, valid_start, valid_stop, nspectra,
                    input_scale
                );
            }
            __syncthreads();
            if (time < native_count && enabled[local_channel]) {
                for (int d = 0; d < dm_count; ++d) {
                    long long delay = delays[(long long)(dm_base + d) * nchans + global_channel];
                    sums[d] += tile[lane + (int)(delay - minimum)];
                }
            }
            __syncthreads();
        }
    } else if (time < native_count) {
        for (int local_channel = 0; local_channel < group_channels; ++local_channel) {
            int global_channel = channel_start + local_channel;
            for (int d = 0; d < dm_count; ++d) {
                long long delay = delays[(long long)(dm_base + d) * nchans + global_channel];
                sums[d] += sample_at(
                    source, medians, enabled, local_channel, output_start + time + delay,
                    source_start, source_count, valid_start, valid_stop, nspectra,
                    input_scale
                );
            }
        }
    }

    if (time < native_count) {
        for (int d = 0; d < dm_count; ++d) {
            long long index = (long long)(dm_base + d) * native_count + time;
            if (!first_group) sums[d] += output[index];
            output[index] = sums[d];
        }
    }
}

#define DECLARE_TYPED_KERNELS(SUFFIX, TYPE) \
extern "C" __global__ void dedisp_ft_##SUFFIX( \
    const TYPE *source, const float *medians, const unsigned char *enabled, \
    const long long *ft_delays, float *output, int nchans, int channel_start, \
    int group_channels, long long source_start, long long source_count, \
    long long valid_start, long long valid_stop, long long nspectra, \
    long long output_start, int native_count, double input_scale \
) { \
    dedisp_ft_body(source, medians, enabled, ft_delays, output, nchans, \
        channel_start, group_channels, source_start, source_count, valid_start, \
        valid_stop, nspectra, output_start, native_count, input_scale); \
} \
extern "C" __global__ void dmt_tiled_##SUFFIX( \
    const TYPE *source, const float *medians, const unsigned char *enabled, \
    const long long *delays, float *output, int nchans, int channel_start, \
    int group_channels, int dmsteps, long long source_start, \
    long long source_count, long long valid_start, long long valid_stop, \
    long long nspectra, long long output_start, int native_count, \
    int shared_capacity, int first_group, double input_scale \
) { \
    extern __shared__ float tile[]; \
    dmt_tiled_body(source, medians, enabled, delays, output, nchans, \
        channel_start, group_channels, dmsteps, source_start, source_count, \
        valid_start, valid_stop, nspectra, output_start, native_count, \
        shared_capacity, first_group, input_scale, tile); \
}

DECLARE_TYPED_KERNELS(u8, uint8_t)
DECLARE_TYPED_KERNELS(u16, uint16_t)
DECLARE_TYPED_KERNELS(i16, int16_t)
DECLARE_TYPED_KERNELS(i32, int32_t)
DECLARE_TYPED_KERNELS(f16, __half)
DECLARE_TYPED_KERNELS(f32, float)
DECLARE_TYPED_KERNELS(f64, double)

extern "C" __global__ void bin_ft(
    const float *native,
    float *output,
    int nchans,
    int native_count,
    int time_size,
    int time_bin
) {
    int time = blockIdx.x * 32 + threadIdx.x;
    int channel = blockIdx.y * 8 + threadIdx.y;
    if (time >= time_size || channel >= nchans) return;
    float sum = 0.0f;
    for (int b = 0; b < time_bin; ++b) {
        sum += native[(long long)(time * time_bin + b) * nchans + channel];
    }
    output[(long long)time * nchans + channel] = sum / time_bin;
}

extern "C" __global__ void bin_dmt(
    const float *native,
    float *output,
    int nchans,
    int dmsteps,
    int native_count,
    int time_size,
    int time_bin
) {
    int time = blockIdx.x * 128 + threadIdx.x;
    int dm = blockIdx.y * 4 + threadIdx.y;
    if (time >= time_size || dm >= dmsteps) return;
    float sum = 0.0f;
    for (int b = 0; b < time_bin; ++b) {
        sum += native[(long long)dm * native_count + time * time_bin + b];
    }
    output[(long long)dm * time_size + time] = sum / ((float)nchans * time_bin);
}
