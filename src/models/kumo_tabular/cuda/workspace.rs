//! Device scratch for one fit or prediction. Each named role grows to its
//! largest use and is overwritten in stream order. No role escapes into a
//! fitted context, and prediction passes share the same workspace.

use cuda_core::{DeviceBuffer, DeviceCopy};

use super::{Device, Precision, device_error};
use crate::Result;
use crate::kernels::cuda::launch::{Activation, Output};

/// Grow without preserving the previous contents, as in `TabICLv2`'s workspace.
pub(super) struct Grow<T: DeviceCopy>(Option<DeviceBuffer<T>>);

impl<T: DeviceCopy> Default for Grow<T> {
    fn default() -> Self {
        Self(None)
    }
}

impl<T: DeviceCopy> Grow<T> {
    pub(super) fn reserve(&mut self, device: &Device, len: usize) -> Result<&mut DeviceBuffer<T>> {
        if self.0.as_ref().is_none_or(|buffer| buffer.len() < len) {
            self.0 = None;
            self.0 = Some(device.zeroed(len).map_err(device_error)?);
        }
        Ok(self.0.as_mut().expect("reserved above"))
    }

    fn get(&self) -> &DeviceBuffer<T> {
        self.0.as_ref().expect("operand was reserved before use")
    }

    /// An exact-length handle for a transfer, cast, or kernel that uses the
    /// buffer length as its shape. Like `slice`, it retains the allocation and
    /// access tracker. Reuse this role only after its last reader is enqueued.
    pub(super) fn prefix(&mut self, device: &Device, len: usize) -> Result<DeviceBuffer<T>>
    where
        T: bytemuck::Pod,
    {
        let buffer = self.reserve(device, len)?;
        let view = buffer.slice(0..len).map_err(device_error)?;
        // SAFETY: this is the initialized, aligned prefix of the same T buffer,
        // not a reinterpretation. T accepts every bit pattern, and both handles
        // use the same allocation and access tracker.
        unsafe { DeviceBuffer::from_storage_copy(view.storage().clone(), len) }.map_err(device_error)
    }
}

#[derive(Default)]
pub(super) struct Operand {
    pub(super) full: Grow<f32>,
    hi: Grow<u16>,
    lo: Grow<u16>,
}

impl Operand {
    pub(super) fn output(&mut self, device: &Device, precision: Precision, len: usize) -> Result<Output<'_>> {
        Ok(match precision {
            Precision::Fp32 => Output::F32(self.full.reserve(device, len)?),
            Precision::Half => Output::Half(self.hi.reserve(device, len)?),
            Precision::Split => Output::Split {
                hi: self.hi.reserve(device, len)?,
                lo: self.lo.reserve(device, len)?,
            },
        })
    }

    pub(super) fn input(&self, precision: Precision) -> Activation<'_> {
        match precision {
            Precision::Fp32 => Activation::F32(self.full.get()),
            Precision::Half => Activation::Half(self.hi.get()),
            Precision::Split => Activation::Split {
                hi: self.hi.get(),
                lo: self.lo.get(),
            },
        }
    }
}

/// The block and head share operands: `a` holds norms and merged attention,
/// `wide` the MLP expansion. The row gate uses `middle` and `projected` after
/// packing queries and before writing the attention residual.
#[derive(Default)]
pub(super) struct Scratch {
    pub(super) a: Operand,
    pub(super) wide: Operand,
    pub(super) projected: Grow<f32>,
    pub(super) middle: Grow<f32>,
    pub(super) packed: Operand,
    pub(super) part: Grow<f32>,
    pub(super) lse: Grow<f32>,
    pub(super) attended: Grow<f32>,
    pub(super) partial: Grow<f32>,
    pub(super) keys: Grow<f32>,
    pub(super) values: Grow<f32>,
    pub(super) half_keys: Grow<u16>,
    pub(super) half_values: Grow<u16>,
}

#[derive(Default)]
pub(super) struct Workspace {
    pub(super) block: Scratch,
    pub(super) input: Grow<f32>,
    pub(super) features: Grow<f32>,
    pub(super) embedded: Grow<f32>,
    pub(super) cells: Grow<f32>,
    pub(super) readouts: Grow<f32>,
    pub(super) sequences: Grow<f32>,
    pub(super) queries: Grow<f32>,
    pub(super) hidden: Grow<f32>,
    pub(super) logits: Grow<f32>,
    pub(super) reduced: Grow<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shorter_pass_reuses_storage_but_exposes_only_its_prefix() {
        let Some(device) = Device::new().unwrap() else {
            return;
        };
        let mut role = Grow::default();
        let mut full = role.prefix(&device, 256).unwrap();
        let first = device.upload(&[1.0]).unwrap();
        let next = device.upload(&[2.0]).unwrap();
        device.repeat(&first, &mut full, (1, 256)).unwrap();
        let mut tail_pass = role.prefix(&device, 17).unwrap();
        assert_eq!(full.cu_deviceptr(), tail_pass.cu_deviceptr());
        let len = tail_pass.len();
        device.repeat(&next, &mut tail_pass, (1, len)).unwrap();
        assert_eq!(device.download(&tail_pass).unwrap(), [2.0; 17]);
        assert_eq!(&device.download(&full).unwrap()[17..], &[1.0; 239]);
    }
}
