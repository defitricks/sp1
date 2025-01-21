use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use cranelift::{
    codegen::{
        self,
        ir::{Block, FuncRef, Function, InstBuilder, Type, Value},
        settings::{self, Configurable},
        CodegenError,
    },
    prelude::AbiParam,
};

use cranelift::frontend;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, ModuleError};
use tracing::instrument;

use crate::{
    events::MemoryRecord, memory::Memory, program, Executor, Instruction, Opcode, Program,
};

mod block;
use block::{BlockBuilder, BuilderResult, JITBlock};
mod builtin;
use builtin::Builtin;

#[cfg(test)]
mod test;
/// A JIT Engine that can run a program in unconstrained mode.
pub struct Engine {
    // The base pc of the program.
    pub(crate) pc_base: u32,

    // The entry points of the program.
    //
    // This is a map of the pc to the compiled function.
    pub(crate) entry_points: BTreeMap<u32, JITBlock>,

    // Cranelift context.
    pub(crate) ctx: codegen::Context,
    // The builder function builder context.
    pub(crate) function_ctx: frontend::FunctionBuilderContext,
    // The JIT module
    pub(crate) jit_module: JITModule,

    #[cfg(test)]
    pub(crate) insert_exit: bool,
}

impl Engine {
    pub fn new(pc_base: u32) -> Engine {
        let isa = cranelift_native::builder().unwrap();

        let mut flags = settings::builder();
        flags.set("is_pic", "false").unwrap();
        flags.set("use_colocated_libcalls", "false").unwrap();

        let builder = JITBuilder::with_isa(
            isa.finish(settings::Flags::new(flags)).unwrap(),
            cranelift_module::default_libcall_names(),
        );

        let jit_module = JITModule::new(builder);

        Self {
            pc_base,
            entry_points: BTreeMap::new(),
            ctx: jit_module.make_context(),
            function_ctx: frontend::FunctionBuilderContext::new(),
            jit_module,
            #[cfg(test)]
            insert_exit: false,
        }
    }

    #[cfg(test)]
    pub fn add_exit(&mut self) {
        self.insert_exit = true;
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct VmContext {
    // A ptr to the registers
    registers: *mut u32,
    // The original memory of the executor
    host_memory: *mut Memory<MemoryRecord>,
    // The ephemeral memory of the executor
    ephemeral_mem: *mut Memory<MemoryRecord>,
    // The target pc of the jump
    pc: u32,
    // 1 if were exiting unconstrained mode.
    exit: u32,
}

impl VmContext {
    fn new(
        registers: *mut u32,
        host_memory: *mut Memory<MemoryRecord>,
        ephemeral_mem: *mut Memory<MemoryRecord>,
        pc: u32,
    ) -> Self {
        Self { registers, host_memory, ephemeral_mem, pc, exit: 0 }
    }
}

impl Engine {
    /// JITs the program starting at the executors current pc.
    ///
    /// The engine will
    ///
    /// Note: Its up to the caller to ensure that we should be using the "unconstrained" JIT Engine.
    pub fn run(&mut self, executor: &mut Executor<'_>) {
        let host_memory = &mut executor.state.memory as *mut Memory<MemoryRecord>;

        let mut ephemeral_mem = Memory::new_preallocated();
        let eph = &mut ephemeral_mem as *mut Memory<MemoryRecord>;

        // Cloning the registers is cheap.
        let mut registers = executor.registers();

        let mut context =
            VmContext::new(registers.as_mut_ptr(), host_memory, eph, executor.state.pc);

        // Start executing the program
        // Our current JIT strategy only compiles up until the next `JALR` instruction.
        loop {
            if let Some(entry_point) = self.entry_points.get(&context.pc) {
                unsafe {
                    entry_point.call(&mut context);
                }

                // We are exiting unconstrained mode.
                if context.exit == 1 {
                    // This is either triggered by a `JALR` or `EXIT_UNCONSTRAINED`
                    executor.state.pc = context.pc;
                    break;
                }

                // If we havent exited, then were handling a JALR.
                debug_assert!(context.pc != 0);
            } else {
                self.compile_and_link(&executor.program, context.pc).expect("failed to compile");
                continue;
            }
        }
    }

    /// Compile the program starting at the given pc until the next `JALR` or `EXIT` syscall.
    ///
    /// Adds the compiled function to the engines `entry_points` map.
    #[instrument(skip(self, program))]
    fn compile_and_link(&mut self, program: &Program, pc: u32) -> Result<(), ModuleError> {
        self.compile(program, pc);

        // Weve finsihed creating the function, now we need to declare it with the JIT module.
        let id = self.jit_module.declare_function(
            &format!("fn_s{pc}"),
            // We want to be able to call this function from outside the JIT module.
            Linkage::Export,
            &self.ctx.func.signature,
        )?;

        // Define the function to jit. This finishes compilation, although
        // there may be outstanding relocations to perform. Currently, jit
        // cannot finish relocations until all functions to be called are
        // defined. For this toy demo for now, we'll just finalize the
        // function below.
        self.jit_module.define_function(id, &mut self.ctx)?;

        // Clear the context state.
        self.jit_module.clear_context(&mut self.ctx);
        self.jit_module.finalize_definitions()?;

        // We can now retrieve a pointer to the machine code.
        let code = self.jit_module.get_finalized_function(id);

        #[allow(clippy::missing_transmute_annotations)]
        self.entry_points.insert(pc, JITBlock { func: unsafe { std::mem::transmute(code) } });

        Ok(())
    }

    /// Compiles the program starting at the given pc until the next `JALR` or `EXIT` syscall.
    ///
    /// The host will call the function being built with the signature `fn(ptr: *mut VmContext)`.
    ///
    /// Returns the pc of the last instruction in the block.
    fn compile(&mut self, program: &Program, pc: u32) {
        // Clear any existing function
        self.ctx.func.clear();

        // The host ptr type.
        let ptr_type = self.jit_module.target_config().pointer_type();

        // We must declare builtins for each individual block.
        let builtins = Self::declare_builtins(&mut self.jit_module, &mut self.ctx.func);

        // A block is a sequence of instructions that are executed in order, until a `JALR` is encountered.
        let mut block_builder =
            BlockBuilder::new(&mut self.ctx.func, &mut self.function_ctx, &builtins, ptr_type);

        // We keep track of branch points, our strategy is to compile them all.
        let mut branch_points: Vec<(u32, Block)> = Vec::new();

        let mut pc = pc;

        loop {
            let idx = ((pc - self.pc_base) / 4) as usize;
            let Some(instruction) = program.instructions.get(idx) else {
                break;
            };

            match block_builder.translate_instruction(instruction, pc) {
                // Everything as normal, just move to the next instruction.
                BuilderResult::Continue | BuilderResult::Auipc => pc += 4,
                // Encounter a branch, add it to the list of branch points to be compiled later.
                BuilderResult::Branch { target, branched, not_branched } => {
                    branch_points.push((pc + 4, not_branched));

                    // We need to compile the branch target now.
                    block_builder.builder.switch_to_block(branched);
                    block_builder.builder.seal_block(branched);

                    // We need to compile the branch target now.
                    pc = target;
                }
                BuilderResult::Jalr | BuilderResult::Ecall => {
                    // In the Ecall case we dont actually know if this is an exit or a write, so we just
                    // assume it is an exit, we can continue on in a new block if its a write

                    // In the JALR case we have no idea where the next instruction is, so we just
                    // continue on in a new block.

                    // In the Exit case we know we are exiting unconstrained mode.

                    // If we have any branch points, we need to compile them now.
                    if let Some((pc_to_continue, block)) = branch_points.pop() {
                        block_builder.builder.switch_to_block(block);
                        block_builder.builder.seal_block(block);

                        pc = pc_to_continue;
                        continue;
                    }

                    // There are no branches and this block is done.
                    break;
                }
                BuilderResult::Jal(next_pc) => {
                    // JAL jumps to immediate pc, so we can just continue translating from there.
                    pc = next_pc;
                }
            }
        }

        #[cfg(test)]
        {
            if self.insert_exit {
                block_builder.exit_unconstrained(pc);
            }
        }

        // Finalize the function
        // Note: This can fail if no return statemnet has been included,
        // This can happy by using assembly or something.
        // Ie. I somehow start this jit at some pc that doesnt have a corresponding syscall or JALR out.
        block_builder.builder.finalize();
    }
}

impl Engine {
    // Declares all the builtints and returns thier FuncRef for the given context / func.
    fn declare_builtins(
        jit_module: &mut JITModule,
        building: &mut Function,
    ) -> BTreeMap<Builtin, FuncRef> {
        let ptr_type = jit_module.target_config().pointer_type();

        let mut builtins = BTreeMap::new();
        for builtin in Builtin::all() {
            let params = builtin.params(ptr_type);
            let returns = builtin.returns();
            let _addr = builtin.addr();
            let name = builtin.name();

            let mut sig = jit_module.make_signature();
            sig.params = params;
            sig.returns = returns;

            let func_id = jit_module
                .declare_function(name, Linkage::Import, &sig)
                .expect("failed to declare builtin function");

            let func = jit_module.declare_func_in_func(func_id, building);

            builtins.insert(builtin, func);
        }

        builtins
    }
}
