/* Any copyright is dedicated to the Public Domain.
 * http://creativecommons.org/publicdomain/zero/1.0/ */

"use strict";

add_task(async function test_webnn_context_loss_after_cycle_collection() {
  await SpecialPowers.pushPrefEnv({ set: [["dom.webnn.enabled", true]] });
  await BrowserTestUtils.withNewTab("https://example.com/", async browser => {
    await SpecialPowers.spawn(browser, [], async () => {
      content.liveWebNNContext = await content.navigator.ml.createContext();
      const context = await content.navigator.ml.createContext();
      const cycle = { context };
      context.lost.then(() => cycle.context);
    });
    await SpecialPowers.spawn(browser, [], () => {
      Cu.forceGC();
      Cu.forceCC();
    });

    await cleanUtilityProcessShutdown("hwInference", true);
    await SpecialPowers.spawn(browser, [], async () => {
      const context = content.liveWebNNContext;
      const lost = await context.lost;
      Assert.ok(lost.message.length, "Process loss resolves the lost promise");
      Assert.throws(
        () => new content.MLGraphBuilder(context),
        /InvalidStateError/,
        "The surviving context cannot build graphs after process loss"
      );
      delete content.liveWebNNContext;
    });
  });
});
