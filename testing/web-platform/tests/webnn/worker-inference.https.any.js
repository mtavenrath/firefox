// META: global=dedicatedworker,sharedworker,serviceworker
// META: timeout=long

'use strict';

promise_test(async t => {
  const context = await navigator.ml.createContext();
  t.add_cleanup(() => context.destroy());
  const descriptor = {dataType: 'float32', shape: [2, 2]};
  const builder = new MLGraphBuilder(context);
  const input = builder.input('input', descriptor);
  const bias = builder.constant(descriptor, new Float32Array([1, 2, 3, 4]));
  const graph = await builder.build({output: builder.relu(builder.add(input, bias))});
  t.add_cleanup(() => graph.destroy());
  const source = await context.createTensor({...descriptor, writable: true});
  const destination = await context.createTensor({...descriptor, readable: true});
  t.add_cleanup(() => source.destroy());
  t.add_cleanup(() => destination.destroy());
  context.writeTensor(source, new Float32Array([-2, 3, -4, 5]));
  context.dispatch(graph, {input: source}, {output: destination});
  assert_array_equals(
      new Float32Array(await context.readTensor(destination)), [0, 5, 0, 9]);
  const output = new Float32Array(4);
  await context.readTensor(destination, output);
  assert_array_equals(output, [0, 5, 0, 9]);
}, 'Workers can build graphs, dispatch computation and read tensors');

promise_test(async t => {
  const context = await navigator.ml.createContext({accelerated: false});
  t.add_cleanup(() => context.destroy());
  const descriptor = {dataType: 'int32', shape: [4]};
  const builder = new MLGraphBuilder(context);
  const input = builder.input('input', descriptor);
  const graph = await builder.build({output: builder.abs(input)});
  t.add_cleanup(() => graph.destroy());
  const source = await context.createTensor({...descriptor, writable: true});
  const destination = await context.createTensor({...descriptor, readable: true});
  t.add_cleanup(() => source.destroy());
  t.add_cleanup(() => destination.destroy());
  context.writeTensor(source, new Int32Array([-2147483647, 0, 2147483646, 2147483647]));
  context.dispatch(graph, {input: source}, {output: destination});
  assert_array_equals(
      new Int32Array(await context.readTensor(destination)),
      [2147483647, 0, 2147483646, 2147483647]);
}, 'Worker tensor transfers preserve exact int32 values');

promise_test(async t => {
  const context = await navigator.ml.createContext();
  const lost = context.lost;
  context.destroy();
  assert_true((await lost).message.length > 0);
  assert_throws_dom('InvalidStateError', () => new MLGraphBuilder(context));
  await promise_rejects_dom(
      t, 'InvalidStateError',
      context.createTensor({dataType: 'float32', shape: [1]}));
}, 'Destroying a worker context resolves its lost promise and prevents new work');
