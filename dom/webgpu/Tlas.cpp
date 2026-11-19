/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "Tlas.h"

#include "Device.h"
#include "mozilla/dom/WebGPUBinding.h"

namespace mozilla::webgpu {
GPU_IMPL_CYCLE_COLLECTION(Tlas, mParent)
GPU_IMPL_JS_WRAP(Tlas)

Tlas::Tlas(Device* aParent, RawId aId)
    : ObjectBase(aParent->GetChild(), aId, ffi::wgpu_client_drop_tlas),
      ChildOf(aParent) {}
Tlas::~Tlas() = default;
}  // namespace mozilla::webgpu
