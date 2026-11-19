importScripts("ray_query_smoke.js");
onmessage = async () => {
  try {
    const wasm = await import("./firefox_ray_query_wasm.js");
    await wasm.default();
    const result = await runRayQuerySmoke(wasm.build_triangle_scene);
    result.optIn = await wasm.wgpu_requires_unsafe_opt_in();
    result.wgpu = [...(await wasm.run_wgpu_ray_query())];
    postMessage(result);
  } catch (error) {
    postMessage({ error: String(error), stack: error.stack });
  }
};
