'use strict';

if (new URL(self.location).searchParams.has('nested')) {
  const worker = new Worker('permissions-policy-worker.js');
  worker.onmessage = event => {
    postMessage(event.data);
    worker.terminate();
    close();
  };
} else {
  navigator.ml.createContext().then(context => {
    context.destroy();
    postMessage('success');
    close();
  }, error => {
    postMessage(error.name);
    close();
  });
}
