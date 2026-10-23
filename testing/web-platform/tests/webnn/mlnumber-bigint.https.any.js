// META: global=window,dedicatedworker
// META: timeout=long

'use strict';

for (const [dataType, ArrayType, maximum] of [
  ['int64', BigInt64Array, (1n << 63n) - 1n],
  ['uint64', BigUint64Array, (1n << 64n) - 1n],
]) {
  async function createContext(t) {
    const context = await navigator.ml.createContext();
    t.add_cleanup(() => context.destroy());
    return context;
  }

  async function readGraph(t, context, builder, output) {
    const graph = await builder.build({output});
    t.add_cleanup(() => graph.destroy());
    const tensor = await context.createTensor({
      dataType,
      shape: output.shape,
      readable: true,
    });
    t.add_cleanup(() => tensor.destroy());
    context.dispatch(graph, {}, {output: tensor});
    return new ArrayType(await context.readTensor(tensor));
  }

  promise_test(async t => {
    const context = await createContext(t);
    if (!context.opSupportLimits().constant.dataTypes.includes(dataType)) {
      return;
    }
    const builder = new MLGraphBuilder(context);
    const output = builder.identity(builder.constant(dataType, maximum - 1n));
    assert_array_equals(await readGraph(t, context, builder, output), [maximum - 1n]);
  }, `${dataType} scalar BigInt constants preserve adjacent large integers`);

  promise_test(async t => {
    const context = await createContext(t);
    if (!context.opSupportLimits().clamp.input.dataTypes.includes(dataType)) {
      return;
    }
    const builder = new MLGraphBuilder(context);
    const input = builder.constant(
        {dataType, shape: [3]}, new ArrayType([maximum - 2n, maximum - 1n, maximum]));
    const output = builder.clamp(input, {minValue: maximum - 1n, maxValue: maximum - 1n});
    assert_array_equals(
        await readGraph(t, context, builder, output),
        [maximum - 1n, maximum - 1n, maximum - 1n]);
  }, `${dataType} clamp BigInt bounds preserve adjacent large integers`);

  promise_test(async t => {
    const context = await createContext(t);
    if (!context.opSupportLimits().clamp.input.dataTypes.includes(dataType)) {
      return;
    }
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {dataType, shape: [1]});
    assert_throws_js(TypeError, () => builder.clamp(
        input, {minValue: maximum, maxValue: maximum - 1n}));
  }, `${dataType} clamp rejects inverted adjacent BigInt bounds`);

  promise_test(async t => {
    const context = await createContext(t);
    if (!context.opSupportLimits().pad.input.dataTypes.includes(dataType)) {
      return;
    }
    const builder = new MLGraphBuilder(context);
    const input = builder.constant({dataType, shape: [1]}, new ArrayType([1n]));
    const output = builder.pad(input, [1], [1], {value: maximum - 1n});
    assert_array_equals(
        await readGraph(t, context, builder, output), [maximum - 1n, 1n, maximum - 1n]);
  }, `${dataType} pad BigInt values preserve adjacent large integers`);
}

for (const method of ['constant', 'clamp', 'pad']) {
  promise_test(async t => {
    const context = await navigator.ml.createContext();
    t.add_cleanup(() => context.destroy());
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {dataType: 'float32', shape: [1]});
    const value = {valueOf() { context.destroy(); return 1; }};
    assert_throws_dom('InvalidStateError', () => {
      if (method === 'constant') {
        builder.constant('float32', value);
      } else if (method === 'clamp') {
        builder.clamp(input, {minValue: value});
      } else {
        builder.pad(input, [1], [1], {value});
      }
    });
  }, `${method} checks context state after converting an MLNumber`);
}
