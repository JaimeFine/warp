import numpy as np
import warp as wp

from warp.tests.interop.test_jax import (
    _import_jax,
    _import_jax_numpy,
    assert_np_equal,
    double_func,
    scale_kernel,
)

def main():
    jax = _import_jax()
    jp = _import_jax_numpy()

    wp.init()
    device = wp.get_device("cpu")
    wp.load_module(module=scale_kernel.module, device=device)

    for graph_mode in (
        wp.JaxCallableGraphMode.WARP_STAGED,
        wp.JaxCallableGraphMode.WARP_STAGED_EX,
    ):
        jax_double = wp.jax_callable(
            double_func,
            graph_mode = graph_mode,
            module_preload_mode = wp.JaxModulePreloadMode.NONE,
        )

        run = jax.jit(
            lambda value, jax_double = jax_double: jax_double(value)[0]
        )

        with jax.default_device(wp.device_to_jax(device)):
            x = jp.arange(32, dtype = jp.float32)
            y = run(x)
            jax.block_until_ready(y)

        assert_np_equal(
            np.asarray(y), 2.0 * np.arange(32, dtype = np.float32)
        )

        call_desc = next(iter(jax_double.call_descriptors.values()))


        print("After first execution:")
        print(f"  call descriptors: \
              {len(jax_double.call_descriptors)}")
        print(f"  graph cache size: \
              {jax_double.graph_cache_size}")
        print(f"  call_desc.capture is None: \
              {call_desc.capture is None}")
        print(f"  input_staging_arrays is None: \
              {call_desc.input_staging_arrays is None}")
        print(f"  output_staging_arrays is None: \
              {call_desc.output_staging_arrays is None}")
        print(f"  static_staging_arrays is None: \
              {call_desc.static_staging_arrays is None}")

        wp.clear_jax_callable_graph_cache(jax_double)

        print("After clear_jax_callable_graph_cache(jax_double):")
        print(f"  graph cache size: \
              {jax_double.graph_cache_size}")
        print(f"  call_desc.capture is None: \
              {call_desc.capture is None}")
        print(f"  input_staging_arrays is None: \
              {call_desc.input_staging_arrays is None}")
        print(f"  output_staging_arrays is None: \
              {call_desc.output_staging_arrays is None}")
        print(f"  static_staging_arrays is None: \
              {call_desc.static_staging_arrays is None}")

if __name__ == "__main__":
    main()