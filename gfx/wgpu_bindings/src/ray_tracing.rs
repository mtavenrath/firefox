/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{client::Client, wgpu_string, DeviceAction, FfiSlice, Message};
use nsstring::nsACString;
use wgpu_core_remote_types::{encoders::CommandEncoderCommand, id, ray_tracing as rt};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FfiTriangleSize {
    vertex_count: u32,
    indexed: bool,
    index_format: wgt::IndexFormat,
    index_count: u32,
    opaque: bool,
}

impl FfiTriangleSize {
    fn to_remote(self) -> rt::TriangleGeometrySize {
        rt::TriangleGeometrySize {
            vertex_count: self.vertex_count,
            index: self.indexed.then_some(rt::TriangleIndexSize {
                format: self.index_format,
                count: self.index_count,
            }),
            opaque: self.opaque,
        }
    }
}

#[repr(C)]
pub struct FfiBlasDescriptor<'a> {
    label: Option<&'a nsACString>,
    geometries: FfiSlice<'a, FfiTriangleSize>,
    fast_build: bool,
}

#[repr(C)]
pub struct FfiTlasDescriptor<'a> {
    label: Option<&'a nsACString>,
    max_instances: u32,
    fast_build: bool,
}

fn preference(fast_build: bool) -> rt::BuildPreference {
    if fast_build {
        rt::BuildPreference::FastBuild
    } else {
        rt::BuildPreference::FastTrace
    }
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_client_create_blas(
    client: &Client,
    device_id: id::DeviceId,
    desc: &FfiBlasDescriptor,
) -> id::BlasId {
    let id = client.identities.lock().blases.process();
    let desc = rt::BlasDescriptor {
        label: wgpu_string(desc.label),
        geometries: unsafe { desc.geometries.as_slice() }
            .iter()
            .map(|s| s.to_remote())
            .collect(),
        build_preference: preference(desc.fast_build),
    };
    client.queue_message(&Message::Device(
        device_id,
        DeviceAction::CreateBlas(id, desc),
    ));
    id
}

#[no_mangle]
pub extern "C" fn wgpu_client_create_tlas(
    client: &Client,
    device_id: id::DeviceId,
    desc: &FfiTlasDescriptor,
) -> id::TlasId {
    let id = client.identities.lock().tlases.process();
    let desc = rt::TlasDescriptor {
        label: wgpu_string(desc.label),
        max_instances: desc.max_instances,
        build_preference: preference(desc.fast_build),
    };
    client.queue_message(&Message::Device(
        device_id,
        DeviceAction::CreateTlas(id, desc),
    ));
    id
}

#[no_mangle]
pub extern "C" fn wgpu_client_drop_blas(client: &Client, id: id::BlasId) {
    client.queue_message(&Message::DropBlas(id));
    client.identities.lock().blases.free(id);
}

#[no_mangle]
pub extern "C" fn wgpu_client_drop_tlas(client: &Client, id: id::TlasId) {
    client.queue_message(&Message::DropTlas(id));
    client.identities.lock().tlases.free(id);
}

#[repr(C)]
pub struct FfiTriangleGeometry {
    size: FfiTriangleSize,
    vertex_buffer: id::BufferId,
    first_vertex: u32,
    vertex_stride: u64,
    index_buffer: Option<id::BufferId>,
    first_index: u32,
}

#[repr(C)]
pub struct FfiBlasBuild<'a> {
    blas: id::BlasId,
    geometries: FfiSlice<'a, FfiTriangleGeometry>,
}

#[repr(C)]
pub struct FfiTlasInstance {
    blas: id::BlasId,
    transform: [f32; 12],
    custom_index: u32,
    mask: u8,
}

#[repr(C)]
pub struct FfiTlasBuild<'a> {
    tlas: id::TlasId,
    instances: FfiSlice<'a, FfiTlasInstance>,
}

#[no_mangle]
pub unsafe extern "C" fn wgpu_client_build_acceleration_structures(
    client: &Client,
    encoder_id: id::CommandEncoderId,
    blases: FfiSlice<'_, FfiBlasBuild>,
    tlases: FfiSlice<'_, FfiTlasBuild>,
) {
    let blases = unsafe { blases.as_slice() }
        .iter()
        .map(|build| rt::BlasBuild {
            blas: build.blas,
            geometries: unsafe { build.geometries.as_slice() }
                .iter()
                .map(|g| rt::TriangleGeometry {
                    size: g.size.to_remote(),
                    vertex_buffer: g.vertex_buffer,
                    first_vertex: g.first_vertex,
                    vertex_stride: g.vertex_stride,
                    index_buffer: g.index_buffer,
                    first_index: g.index_buffer.map(|_| g.first_index),
                })
                .collect(),
        })
        .collect();
    let tlases = unsafe { tlases.as_slice() }
        .iter()
        .map(|build| rt::TlasBuild {
            tlas: build.tlas,
            instances: unsafe { build.instances.as_slice() }
                .iter()
                .map(|i| rt::TlasInstance {
                    blas: i.blas,
                    transform: i.transform,
                    custom_index: i.custom_index,
                    mask: i.mask,
                })
                .collect(),
        })
        .collect();
    let command = CommandEncoderCommand::BuildAccelerationStructures { blases, tlases };
    client.queue_message(&Message::CommandEncoder(encoder_id, command));
}
