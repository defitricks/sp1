//! Builtin functions for the JIT compiler.
//!
//! We want to use an ephermal memory setup so we branch on our first reads to the memory.
//!
//! We also need to maintain checkpoints when were in checkpointing mode.

use super::VmContext;

extern "C" fn load(ctx: *mut VmContext, address: u32, data: u32) {}

extern "C" fn store(ctx: *mut VmContext, address: u32, data: u32) {}

extern "C" fn handle_syscall(ctx: *mut VmContext, syscall_id: u32, a: u32, b: u32) {}

extern "C" fn write_to_input_stream(ctx: *mut VmContext, ptr: *mut u8, len: u32) {}

unsafe fn checkpoint(ctx: *mut VmContext) {
    todo!()
}
