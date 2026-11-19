export async function renderTriangle(device, width, height) {
  device.pushErrorScope("validation");
  const vertexBuffer = device.createBuffer({
    size: 36,
    usage: GPUBufferUsage.BLAS_INPUT | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    vertexBuffer,
    0,
    new Float32Array([-1, -1, 2, 1, -1, 2, 0, 1, 2])
  );
  const blas = device.createBlas({ geometries: [{ vertexCount: 3 }] });
  const tlas = device.createTlas({ maxInstances: 1 });
  const size = width * height * 4;
  const output = device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const module = device.createShaderModule({
    code: `
    enable wgpu_ray_query;
    @group(0) @binding(0) var scene: acceleration_structure;
    @group(0) @binding(1) var<storage, read_write> pixels: array<u32>;
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) id: vec3u) {
      if (id.x >= ${width}u || id.y >= ${height}u) { return; }
      let position = (vec2f(id.xy) + 0.5) / vec2f(${width}.0, ${height}.0);
      let direction = normalize(vec3f((position.x - 0.5) * 3.0,
                                     (0.5 - position.y) * 3.0, 2.0));
      var query: ray_query;
      rayQueryInitialize(&query, scene,
        RayDesc(0u, 255u, 0.001, 100.0, vec3f(0.0), direction));
      while rayQueryProceed(&query) {}
      let hit = rayQueryGetCommittedIntersection(&query);
      var color = vec3u(18u, 26u, 43u);
      if (hit.kind == 1u) {
        let bary = hit.barycentrics;
        color = vec3u(clamp(vec3f(1.0 - bary.x - bary.y, bary.x, bary.y),
                           vec3f(0.0), vec3f(1.0)) * 255.0);
      }
      pixels[id.y * ${width}u + id.x] =
        color.x | (color.y << 8u) | (color.z << 16u) | 0xff000000u;
    }
  `,
  });
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: tlas },
      { binding: 1, resource: { buffer: output } },
    ],
  });
  const encoder = device.createCommandEncoder();
  encoder.buildAccelerationStructures({
    blasBuilds: [{ blas, geometries: [{ vertexBuffer, vertexCount: 3 }] }],
    tlasBuilds: [
      {
        tlas,
        instances: [{ blas, transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0] }],
      },
    ],
  });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, group);
  pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
  pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const pixels = new Uint8ClampedArray(readback.getMappedRange()).slice();
  readback.unmap();
  const error = await device.popErrorScope();
  vertexBuffer.destroy();
  output.destroy();
  readback.destroy();
  if (error) {
    throw error;
  }
  return pixels;
}
