/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

async function runRayQuerySmoke(buildScene) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter?.features.has("wgpu-ray-query")) {
    return { skipped: "No ray-query capable adapter" };
  }
  const device = await adapter.requestDevice({
    requiredFeatures: ["wgpu-ray-query"],
  });
  const checks = [];
  const check = (value, message) => checks.push({ value: !!value, message });
  device.addEventListener("uncapturederror", event => {
    check(false, "Uncaptured GPU error: " + event.error.message);
  });
  const vertices = device.createBuffer({
    size: 36,
    usage: GPUBufferUsage.BLAS_INPUT | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(
    vertices,
    0,
    new Float32Array([-1, -1, 2, 1, -1, 2, 0, 1, 2])
  );
  const indices = device.createBuffer({
    size: 12,
    usage: GPUBufferUsage.BLAS_INPUT | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(indices, 0, new Uint32Array([0, 1, 2]));

  const shader = device.createShaderModule({
    code: `
    enable wgpu_ray_query;
    @group(0) @binding(0) var scene: acceleration_structure;
    @group(0) @binding(1) var<storage, read_write> result: array<vec4u>;
    @compute @workgroup_size(4)
    fn main(@builtin(local_invocation_index) id: u32) {
      var query: ray_query;
      let origin = select(vec3f(0.0), vec3f(3.0, 0.0, 0.0), id == 1u);
      let mask = select(1u, 2u, id == 2u);
      let maxT = select(10.0, 1.0, id == 3u);
      rayQueryInitialize(&query, scene,
        RayDesc(0u, mask, 0.001, maxT, origin, vec3f(0.0, 0.0, 1.0)));
      while rayQueryProceed(&query) {}
      let hit = rayQueryGetCommittedIntersection(&query);
      result[id] = vec4u(hit.kind, bitcast<u32>(hit.t),
                        hit.instance_custom_data, hit.primitive_index);
    }
  `,
  });
  const layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        accelerationStructure: {},
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
  });
  const pipeline = await device.createComputePipelineAsync({
    layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module: shader, entryPoint: "main" },
  });

  const cases = [
    { indexed: false, depth: 0 },
    { indexed: true, depth: 0 },
  ];
  if (!buildScene) {
    cases.push({ indexed: false, depth: 1 });
    cases.push({ indexed: false, depth: 0, release: true });
  }
  for (const { indexed, depth, release } of cases) {
    device.pushErrorScope("validation");
    const size = { vertexCount: 3 };
    if (indexed) {
      size.indexFormat = "uint32";
      size.indexCount = 3;
    }
    const encoder = device.createCommandEncoder();
    let blas = buildScene
      ? null
      : device.createBlas({ label: "triangle", geometries: [size] });
    let tlas = buildScene
      ? buildScene(device, encoder, vertices, indexed ? indices : undefined)
      : device.createTlas({ label: "scene", maxInstances: 1 });
    const output = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readback = device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    let group = device.createBindGroup({
      layout,
      entries: [
        { binding: 0, resource: tlas },
        { binding: 1, resource: { buffer: output } },
      ],
    });
    const geometry = { ...size, vertexBuffer: vertices };
    if (indexed) {
      geometry.indexBuffer = indices;
    }
    if (!buildScene) {
      encoder.buildAccelerationStructures({
        blasBuilds: [{ blas, geometries: [geometry] }],
        tlasBuilds: [
          {
            tlas,
            instances: [
              {
                blas,
                customIndex: 7,
                mask: 1,
                transform: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, depth],
              },
            ],
          },
        ],
      });
    }
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, 64);
    const commands = encoder.finish();
    if (release) {
      blas = null;
      tlas = null;
      group = null;
      if (typeof SpecialPowers !== "undefined") {
        SpecialPowers.forceGC();
        SpecialPowers.forceCC();
      }
    }
    device.queue.submit([commands]);
    await readback.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(readback.getMappedRange());
    const distances = new Float32Array(data.buffer);
    check(data[0] === 1, `${indexed}: center ray hits the triangle`);
    check(
      Math.abs(distances[1] - (2 + depth)) < 0.0001,
      `${indexed}/${depth}: transformed hit distance is correct`
    );
    check(data[2] === 7, `${indexed}: custom instance index survives`);
    check(data[3] === 0, `${indexed}: primitive index is zero`);
    check(data[4] === 0, `${indexed}: off-triangle ray misses`);
    check(data[8] === 0, `${indexed}: instance mask rejects the ray`);
    check(data[12] === 0, `${indexed}: maximum ray distance rejects the hit`);
    readback.unmap();
    check(
      (await device.popErrorScope()) === null,
      `${indexed}: valid build and dispatch`
    );
    output.destroy();
    readback.destroy();
  }
  vertices.destroy();
  indices.destroy();
  device.destroy();
  check(
    (await device.lost).reason === "destroyed",
    "Explicit device loss is reported"
  );
  return { checks };
}

if (typeof WorkerGlobalScope !== "undefined") {
  onmessage = async () => {
    try {
      postMessage(await runRayQuerySmoke());
    } catch (error) {
      postMessage({ error: String(error), stack: error.stack });
    }
  };
}
