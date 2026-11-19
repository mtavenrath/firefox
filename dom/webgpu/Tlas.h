/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef GPU_Tlas_H_
#define GPU_Tlas_H_

#include "ObjectModel.h"

namespace mozilla::webgpu {
class Device;

class Tlas final : public nsWrapperCache,
                   public ObjectBase,
                   public ChildOf<Device> {
 public:
  GPU_DECL_CYCLE_COLLECTION(Tlas)
  GPU_DECL_JS_WRAP(Tlas)

  Tlas(Device* aParent, RawId aId);

 private:
  ~Tlas() override;
};
}  // namespace mozilla::webgpu
#endif
