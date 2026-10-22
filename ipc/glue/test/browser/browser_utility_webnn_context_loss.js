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

add_task(async function test_webnn_worker_shutdown() {
  await SpecialPowers.pushPrefEnv({ set: [["dom.webnn.enabled", true]] });
  for (const action of ["terminate", "close"]) {
    await BrowserTestUtils.withNewTab("https://example.com/", async browser => {
      await SpecialPowers.spawn(browser, [], async () => {
        const source = `
          self.context = await navigator.ml.createContext();
          onmessage = () => {
            navigator.ml.createContext().catch(() => {});
            close();
          };
          postMessage('ready');
        `;
        const url = content.URL.createObjectURL(
          new content.Blob([source], { type: "text/javascript" })
        );
        const worker = new content.Worker(url, { type: "module" });
        content.webnnLifecycleWorker = worker;
        content.webnnLifecycleURL = url;
        await new Promise((resolve, reject) => {
          worker.onmessage = resolve;
          worker.onerror = event => reject(new Error(event.message));
        });
      });
      const pid = await checkUtilityExists("hwInference");
      const shutdown = TestUtils.topicObserved(
        "ipc:utility-shutdown",
        (subject, data) => Number(data) === pid
      );
      await SpecialPowers.spawn(browser, [action], shutdownAction => {
        const worker = content.webnnLifecycleWorker;
        if (shutdownAction === "terminate") {
          worker.terminate();
        } else {
          worker.postMessage("close");
        }
        content.URL.revokeObjectURL(content.webnnLifecycleURL);
        delete content.webnnLifecycleWorker;
        delete content.webnnLifecycleURL;
      });
      const [subject] = await shutdown;
      Assert.ok(
        !subject.QueryInterface(Ci.nsIPropertyBag2).hasKey("dumpID"),
        `${action} releases the last inference connection without crashing`
      );
    });
  }
});
