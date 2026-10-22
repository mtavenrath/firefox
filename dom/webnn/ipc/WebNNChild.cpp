/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "mozilla/webnn/WebNNChild.h"

#include "mozilla/dom/ContentChild.h"
#include "mozilla/dom/MLContext.h"
#include "mozilla/dom/WorkerPrivate.h"
#include "mozilla/dom/WorkerRef.h"
#include "nsThreadUtils.h"
#include "nsXULAppAPI.h"

namespace mozilla::webnn {

WebNNChild::WebNNChild(dom::MLContext* aContext) : mContext(aContext) {}

WebNNChild::~WebNNChild() = default;

#ifndef ANDROID
static bool CreateWebNNParent(ipc::Endpoint<PWebNNParent>&& aEndpoint) {
  MOZ_ASSERT(NS_IsMainThread());
  dom::ContentChild* content = dom::ContentChild::GetSingleton();
  return content && content->SendAcquireHWInferenceProcess() &&
         content->SendCreateWebNN(std::move(aEndpoint));
}

static void ReleaseHWInferenceConnection() {
  MOZ_ASSERT(NS_IsMainThread());
  if (dom::ContentChild* content = dom::ContentChild::GetSingleton()) {
    content->SendReleaseHWInferenceConnection();
  }
}
#endif

/* static */
already_AddRefed<WebNNChild> WebNNChild::Create(dom::MLContext* aContext) {
#ifdef ANDROID
  return nullptr;
#else
  if (!XRE_IsContentProcess()) {
    return nullptr;
  }
  RefPtr<WebNNChild> child = new WebNNChild(aContext);
  dom::WorkerPrivate* worker = nullptr;
  if (!NS_IsMainThread()) {
    worker = dom::GetCurrentThreadWorkerPrivate();
    MOZ_ASSERT(worker);
    child->mWorkerRef = dom::IPCWorkerRef::Create(
        worker, "WebNNChild", [child]() { child->Detach(); });
    if (!child->mWorkerRef) {
      return nullptr;
    }
  }

  ipc::Endpoint<PWebNNParent> parentEndpoint;
  ipc::Endpoint<PWebNNChild> childEndpoint;
  if (NS_FAILED(PWebNN::CreateEndpoints(&parentEndpoint, &childEndpoint)) ||
      !childEndpoint.Bind(child)) {
    child->mWorkerRef = nullptr;
    return nullptr;
  }

  if (worker) {
    nsresult rv = NS_DispatchToMainThread(NS_NewRunnableFunction(
        "WebNNChild::CreateWebNNParent",
        [endpoint = std::move(parentEndpoint)]() mutable {
          CreateWebNNParent(std::move(endpoint));
        }));
    if (NS_FAILED(rv)) {
      child->Detach();
      return nullptr;
    }
    child->mHoldsConnection = true;
  } else {
    child->mHoldsConnection = true;
    if (!CreateWebNNParent(std::move(parentEndpoint))) {
      child->Detach();
      return nullptr;
    }
  }
  return child.forget();
#endif
}

void WebNNChild::Detach() {
  mContext = nullptr;
  if (CanSend()) {
    Close();
  }
}

mozilla::ipc::IPCResult WebNNChild::RecvContextLost(const nsCString& aMessage) {
  if (mContext) {
    RefPtr<dom::MLContext> context = mContext;
    context->OnContextLost(aMessage);
  }
  return IPC_OK();
}

void WebNNChild::ActorDestroy(ActorDestroyReason aReason) {
#ifndef ANDROID
  if (mHoldsConnection) {
    mHoldsConnection = false;
    if (NS_IsMainThread()) {
      ReleaseHWInferenceConnection();
    } else {
      MOZ_ALWAYS_SUCCEEDS(NS_DispatchToMainThread(
          NS_NewRunnableFunction("WebNNChild::ReleaseHWInferenceConnection",
                                 []() { ReleaseHWInferenceConnection(); })));
    }
  }
#endif
  if (mContext) {
    RefPtr<dom::MLContext> context = mContext;
    mContext = nullptr;
    context->OnContextLost("The WebNN process went away"_ns);
  }
  mWorkerRef = nullptr;
}

}  // namespace mozilla::webnn
