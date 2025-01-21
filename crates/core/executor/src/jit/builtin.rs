//! Builtin functions for the JIT compiler.
//!
//! We want to use an ephermal memory setup so we branch on our first reads to the memory.
//!
//! We also need to maintain checkpoints when were in checkpointing mode.

use crate::{events::MemoryRecord, memory::Memory};

use super::VmContext;

use cranelift::{
    codegen::ir::types::{Type, I32},
    prelude::AbiParam,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) enum Builtin {
    Load,
    Store,
    HandleSyscall,
}

impl Builtin {
    pub(crate) const fn all() -> [Self; 3] {
        [Self::Load, Self::Store, Self::HandleSyscall]
    }

    pub(crate) fn params(&self, host_ptr: Type) -> Vec<AbiParam> {
        match self {
            Self::Load => vec![
                // Host memory pointer
                AbiParam::new(host_ptr),
                // Ephemeral memory pointer
                AbiParam::new(host_ptr),
                // The address to load from
                AbiParam::new(I32),
            ],
            Self::Store => vec![
                // Ephemeral memory pointer
                AbiParam::new(host_ptr),
                // The address to store to
                AbiParam::new(I32),
                // The data to store
                AbiParam::new(I32),
            ],
            Self::HandleSyscall => {
                vec![
                    // The input stream ptr
                    AbiParam::new(host_ptr),
                    // The syscall id
                    AbiParam::new(I32),
                    // The first argument
                    AbiParam::new(I32),
                    // The second argument
                    AbiParam::new(I32),
                ]
            }
        }
    }

    pub(crate) fn returns(&self) -> Vec<AbiParam> {
        match self {
            Self::Load => vec![AbiParam::new(I32)],
            Self::Store | Self::HandleSyscall => vec![],
        }
    }

    pub(crate) fn name(&self) -> &str {
        match self {
            Self::Load => "__jit__load",
            Self::Store => "__jit__store",
            Self::HandleSyscall => "__jit__handle_syscall",
        }
    }

    pub(crate) fn addr(&self) -> usize {
        match self {
            Self::Load => __jit__load as usize,
            Self::Store => __jit__store as usize,
            Self::HandleSyscall => __jit__handle_syscall as usize,
        }
    }
}

/// SAFTEY: The function is called by the JIT compiler,
/// So we must assume all pointers and data is valid.
///
/// Loads a word from the host memory or the ephemeral memory and returns it as a value.
#[no_mangle]
extern "C" fn __jit__load(
    host: *mut Memory<MemoryRecord>,
    ephemeral: *mut Memory<MemoryRecord>,
    address: u32,
) -> u32 {
    debug_assert!(address % 4 == 0);

    let host = unsafe { &mut *host };
    let eph = unsafe { &mut *ephemeral };

    if let Some(record) = eph.get(address) {
        // We already have a value read during unconstrained,
        // So now we just need to insert it into memory
        record.value
    } else if let Some(record) = host.get(address) {
        let value = record.value;
        // Weve read from host memory, so we need to insert it into ephemeral memory
        eph.insert(address, *record);

        value
    } else {
        // Weve read from host memory, but this address is not in memory.
        // So we need to insert a zero value into ephemeral memory
        eph.insert(address, MemoryRecord { value: 0, ..Default::default() });
        0
    }
}

/// SAFTEY: The function is called by the JIT compiler,
/// So we must assume all pointers and data is valid.
///
/// Stores a word into the ephemeral memory.
#[no_mangle]
extern "C" fn __jit__store(ephemeral: *mut Memory<MemoryRecord>, address: u32, data: u32) {
    let eph = unsafe { &mut *ephemeral };

    eph.insert(address, MemoryRecord { value: data, ..Default::default() });
}

/// SAFTEY: The function is called by the JIT compiler,
/// So we must assume all pointers and data is valid.
///
/// Handles a syscall.
#[no_mangle]
#[no_mangle]
extern "C" fn __jit__handle_syscall(ctx: *mut VmContext, syscall_id: u32, a: u32, b: u32) {}
