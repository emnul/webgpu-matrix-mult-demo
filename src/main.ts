async function accessGPU(): Promise<GPUDevice | undefined> {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return;
  }
  return await adapter.requestDevice();
}

// function writeBufferMemory(device: GPUDevice) {
//   // Get a GPU buffer in a mapped state and an arrayBuffer for writing.
//   const gpuBuffer = device.createBuffer({
//     mappedAtCreation: true,
//     size: 4,
//     usage: GPUBufferUsage.MAP_WRITE,
//   });
//   console.log(gpuBuffer);
//   const arrayBuffer = gpuBuffer.getMappedRange();
//   console.log(arrayBuffer);
//   // Write bytes to buffer.
//   new Uint8Array(arrayBuffer).set([1, 0, 0, 0]);
// }

// async function copyBuffer(device: GPUDevice) {
//   const gpuWriteBuffer = device.createBuffer({
//     size: 4,
//     usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
//     mappedAtCreation: true,
//   });

//   // When mapped, GPU buffers can be read and written in JavaScript.
//   const arrayBuffer = gpuWriteBuffer.getMappedRange();

//   // Write to arrayBuffer
//   new Uint8Array(arrayBuffer).set([1, 0, 0, 0]);

//   // GPU buffers have to be unmapped to be used in device queue submission.
//   gpuWriteBuffer.unmap();

//   // create new buffer to copy to and read from
//   const gpuReadBuffer = device.createBuffer({
//     size: 4,
//     usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
//   });

//   // Encode copy commands
//   const copyEncoder = device.createCommandEncoder();
//   copyEncoder.copyBufferToBuffer(gpuWriteBuffer, 0, gpuReadBuffer, 0, 4);

//   // Submit commands
//   const gpuCommands = copyEncoder.finish();
//   device.queue.submit([gpuCommands]);

//   // Wait until we can read contents of GPUBuffer
//   await gpuReadBuffer.mapAsync(GPUMapMode.READ);
//   const copiedArray = gpuReadBuffer.getMappedRange();
//   console.log({ copiedArray });
// }

async function matrixMultiplicationGPU(
  device: GPUDevice,
  matrix1: Float32Array,
  matrix2: Float32Array
) {
  // We need to store and retrieve data of matrices in the compute shader
  const matrixGpuBuffer1 = device.createBuffer({
    size: matrix1.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const matrixArrayBuffer1 = matrixGpuBuffer1.getMappedRange();
  new Float32Array(matrixArrayBuffer1).set(matrix1);
  matrixGpuBuffer1.unmap();

  const matrixGpuBuffer2 = device.createBuffer({
    size: matrix2.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const matrixArrayBuffer2 = matrixGpuBuffer2.getMappedRange();
  new Float32Array(matrixArrayBuffer2).set(matrix2);
  matrixGpuBuffer2.unmap();

  // Result matrix
  // n x m * m x n array = n x n result array + 2 to account for row / col size slots
  const resultMatrixBuffSize =
    Float32Array.BYTES_PER_ELEMENT * (2 + matrix1[0] * matrix2[1]);
  const finalMatrixGpuBuffer = device.createBuffer({
    size: resultMatrixBuffSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // A bind group layout defines the input/output interface expected by a shader
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
    ],
  });

  // A bind group defines the actual input/output data expected by the shader
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: matrixGpuBuffer1,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: matrixGpuBuffer2,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: finalMatrixGpuBuffer,
        },
      },
    ],
  });

  const shaderModule = device.createShaderModule({
    code: `
    struct Matrix {
      size : vec2f,
      numbers: array<f32>,
    }

    @group(0) @binding(0) var<storage, read> firstMatrix : Matrix;
    @group(0) @binding(1) var<storage, read> secondMatrix : Matrix;
    @group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;

    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id : vec3u) {
      // Guard against out-of-bounds work group sizes
      if (global_id.x >= u32(firstMatrix.size.x) || global_id.y >= u32(secondMatrix.size.y)) {
        return;
      }

      resultMatrix.size = vec2(firstMatrix.size.x, secondMatrix.size.y);

      let resultCell = vec2(global_id.x, global_id.y);
      var result = 0.0;
      for (var i = 0u; i < u32(firstMatrix.size.y); i = i + 1u) {
        let a = i + resultCell.x * u32(firstMatrix.size.y);
        let b = resultCell.y + i * u32(secondMatrix.size.y);
        result = result + firstMatrix.numbers[a] * secondMatrix.numbers[b];
      }

      let index = resultCell.y + resultCell.x * u32(secondMatrix.size.y);
      resultMatrix.numbers[index] = result;
    }
  `,
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout],
    }),
    compute: {
      module: shaderModule,
      entryPoint: "main",
    },
  });

  const commandEncoder = device.createCommandEncoder();

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  const workgroupCountX = Math.ceil(matrix1[0] / 8);
  const workgroupCountY = Math.ceil(matrix2[1] / 8);
  passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY);
  passEncoder.end();

  const gpuReadBuffer = device.createBuffer({
    size: resultMatrixBuffSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  commandEncoder.copyBufferToBuffer(
    finalMatrixGpuBuffer,
    0,
    gpuReadBuffer,
    0,
    resultMatrixBuffSize
  );
  // Submit commands
  const gpuCommands = commandEncoder.finish();
  device.queue.submit([gpuCommands]);

  // Read result matrix
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  const readArrayBuffer = gpuReadBuffer.getMappedRange();
}

function matrixMultiplicationCPU(
  a: Float32Array,
  b: Float32Array
): Float32Array {
  // Extract the size of the matrices (n x n) from the first two elements
  const n: number = a[0];
  // Check if the matrices are of compatible size
  if (
    a.length !== n * n + 2 ||
    b.length !== n * n + 2 ||
    a[0] !== b[0] ||
    a[1] !== b[1]
  ) {
    throw new Error("Invalid matrix sizes or format.");
  }

  // Initialize the result matrix as a flattened array
  const result = new Float32Array(n * n + 2);
  result[0] = n; // Set the first two elements to n
  result[1] = n;

  // Perform the matrix multiplication
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        // Adjust indices to account for the first two elements in the arrays
        sum += a[2 + row * n + k] * b[2 + k * n + col];
      }
      result[2 + row * n + col] = sum;
    }
  }

  return result;
}

// listen for form submission
document
  .getElementById("form")
  ?.addEventListener("submit", async (event): Promise<void> => {
    event.preventDefault();
    const selectElem = document.getElementById("options") as HTMLSelectElement;

    const matrixSize = Number(selectElem.value);

    const device = await accessGPU();

    const matrix1 = new Float32Array(matrixSize ** 2 + 2);
    const matrix2 = new Float32Array(matrixSize ** 2 + 2);
    // init array elements
    matrix1[0] = matrixSize;
    matrix1[1] = matrixSize;
    matrix2[0] = matrixSize;
    matrix2[1] = matrixSize;
    for (let i = 2; i < matrix1.length; i++) {
      matrix1[i] = 2 * Math.random() + 1;
      matrix2[i] = 2 * Math.random() + 1;
    }

    if (device) {
      // writeBufferMemory(device);
      // copyBuffer(device);
      const start = performance.now();
      await matrixMultiplicationGPU(device, matrix1, matrix2);
      const end = performance.now();
      const computeTime = (end - start).toFixed(2);
      const gpuDiv = document.getElementById("gpu-time") as HTMLDivElement;
      gpuDiv.innerText = computeTime + "ms";
    }

    const start = performance.now();
    matrixMultiplicationCPU(matrix1, matrix2);
    const end = performance.now();
    const computeTime = (end - start).toFixed(2);
    const cpuDiv = document.getElementById("cpu-time") as HTMLDivElement;
    cpuDiv.innerText = computeTime + "ms";
  });

export {};
