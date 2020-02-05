import argparse
import ctypes
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

class DeviceBuffer(object):
    def __init__(self, shape, dtype=trt.int32):
        self.buf = cuda.mem_alloc(trt.volume(shape) * dtype.itemsize)

    def binding(self):
        return int(self.buf)

    def free(self):
        self.buf.free()


def main():
    parser = argparse.ArgumentParser(description='BERT Inference Benchmark')
    parser.add_argument("-e", "--engine", help='Path to BERT TensorRT engine')
    parser.add_argument('-b', '--batch-size', default=[], action="append", help='Batch size(s) to benchmark. Can be specified multiple times for more than one batch size. This script assumes that the engine has been built with one optimization profile for each batch size, and that these profiles are in order of increasing batch size.', type=int)
    parser.add_argument('-s', '--sequence-length', default=128, help='Sequence length of the BERT model', type=int)
    parser.add_argument('-i', '--iterations', default=200, help='Number of iterations to run when benchmarking each batch size.', type=int)
    parser.add_argument('-w', '--warm-up-runs', default=10, help='Number of iterations to run prior to benchmarking.', type=int)
    parser.add_argument('-r', '--random-seed', required=False, default=12345, help='Random seed.', type=int)
    args, _ = parser.parse_known_args()
    args.batch_size = args.batch_size or [1]

    # Import necessary plugins for BERT TensorRT
    ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)

    # The very first context created will use profile 0 by default. We must use this context for profile 0.
    with open(args.engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as default_context:

        # Allocate buffers large enough to store the largest batch size
        max_input_shape = (args.sequence_length, max(args.batch_size))
        max_output_shape = (args.sequence_length, max(args.batch_size), 2, 1, 1)
        buffers = [
            DeviceBuffer(max_input_shape),
            DeviceBuffer(max_input_shape),
            DeviceBuffer(max_input_shape),
            DeviceBuffer(max_output_shape)
        ]

        # Prepare random input
        pseudo_vocab_size = 30522
        pseudo_type_vocab_size = 2
        np.random.seed(args.random_seed)
        test_word_ids = np.random.randint(0, pseudo_vocab_size, (args.sequence_length, max(args.batch_size)), dtype=np.int32)
        test_segment_ids = np.random.randint(0, pseudo_type_vocab_size, (args.sequence_length, max(args.batch_size)), dtype=np.int32)
        test_input_mask = np.ones((args.sequence_length, max(args.batch_size)), dtype=np.int32)

        # Copy input h2d
        cuda.memcpy_htod(buffers[0].buf, test_word_ids.ravel())
        cuda.memcpy_htod(buffers[1].buf, test_segment_ids.ravel())
        cuda.memcpy_htod(buffers[2].buf, test_input_mask.ravel())

        num_binding_per_profile = engine.num_bindings // engine.num_optimization_profiles

        bench_times = {}

        for idx, batch_size in enumerate(sorted(args.batch_size)):
            context = engine.create_execution_context() if idx else default_context
            context.active_optimization_profile = idx

            # Each profile has unique bindings
            binding_idx_offset = idx * num_binding_per_profile
            bindings = [0] * binding_idx_offset + [buf.binding() for buf in buffers]

            shapes = {
                "input_ids": (args.sequence_length, batch_size),
                "segment_ids": (args.sequence_length, batch_size),
                "input_mask": (args.sequence_length, batch_size),
            }

            for binding, shape in shapes.items():
                context.set_binding_shape(engine[binding] + binding_idx_offset, shape)
            assert context.all_binding_shapes_specified

            # Inference
            total_time = 0
            start = cuda.Event()
            end = cuda.Event()
            stream = cuda.Stream()

            # Warmup
            for _ in range(args.warm_up_runs):
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                stream.synchronize()

            # Timing loop
            times = []
            for _ in range(args.iterations):
                start.record(stream)
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                end.record(stream)
                stream.synchronize()
                times.append(end.time_since(start))

            # Compute average time, 95th percentile time and 99th percentile time.
            bench_times[batch_size] = times

            # Clean up each context besides the default context
            if idx:
                context.__del__()

        [b.free() for b in buffers]

        for batch_size, times in bench_times.items():
            total_time = sum(times)
            avg_time = total_time / float(len(times))
            times.sort()
            percentile95 = times[int(len(times) * 0.95)]
            percentile99 = times[int(len(times) * 0.99)]
            print("Running {:} iterations with Batch Size: {:}\n\tTotal Time: {:} ms \tAverage Time: {:} ms\t95th Percentile Time: {:} ms\t99th Percentile Time: {:}".format(args.iterations, batch_size, total_time, avg_time, percentile95, percentile99))


if __name__ == '__main__':
    main()
