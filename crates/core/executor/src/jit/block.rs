use std::{collections::BTreeMap, mem::offset_of};

use cranelift::{
    codegen::{
        self,
        ir::{condcodes::IntCC, AbiParam, FuncRef, Function, InstBuilder, MemFlags, Type, Value},
        settings::{self, Configurable},
        CodegenError,
    },
    prelude::FunctionBuilderContext,
};

use cranelift::frontend;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module, ModuleError};

use crate::{
    events::MemoryRecord, memory::Memory, Executor, Instruction, Opcode, Program, Register,
};

use super::{Block, Builtin, VmContext};

/// A block of code to be executed.
pub struct JITBlock {
    // The actual function that will be executed
    pub func: fn(
        registers: *mut u32,
        host_memory: *mut Memory<MemoryRecord>,
        ephemeral_mem: *mut Memory<MemoryRecord>,
        pc: *mut u32,
        exit: *mut u32,
    ),
}

impl JITBlock {
    pub unsafe fn call(&self, context: &mut VmContext) {
        let VmContext { registers, host_memory, ephemeral_mem, pc, exit } = context;

        (self.func)(*registers, *host_memory, *ephemeral_mem, pc, exit);
    }
}

/// A block builder translates risc32 instructions into a cranelift function.
///
/// The function has 4 parameters:
/// - Registers `*mut u32`
/// - Host Memory `*mut Memory<MemoryRecord>`
/// - Ephemeral Memory `*mut Memory<MemoryRecord>`
/// - Pc `*mut u32`
/// - Exit `*mut u32`
pub struct BlockBuilder<'b> {
    /// The builder context for the JIT
    pub(super) builder: frontend::FunctionBuilder<'b>,

    /// The builtins that are available to the block
    builtins: &'b BTreeMap<Builtin, FuncRef>,

    /// The native pointer type of the host.
    ptr_type: Type,

    /// The pointers to the host state
    registers_ptr: Value,
    pc_ptr: Value,
    exit_ptr: Value,
    host_memory_ptr: Value,
    ephemeral_mem_ptr: Value,

    /// Types that represent the formal types of the ISA
    ///
    /// These will be used for both signed and unsigned types.
    int_8_type: Type,
    int_16_type: Type,
    int_32_type: Type,
    // Used only for certian ALU instructions
    int_64_type: Type,
}

impl<'b> BlockBuilder<'b> {
    /// Sets up the entry block and gets the context pointer.
    pub fn new(
        func: &'b mut Function,
        ctx: &'b mut FunctionBuilderContext,
        builtins: &'b BTreeMap<Builtin, FuncRef>,
        ptr_type: Type,
    ) -> Self {
        // First setup the function signature
        // We know were going to need:
        // - Registers
        // - Host Memory
        // - Ephemeral Memory
        // - Pc
        // - Exit

        // Registers
        func.signature.params.push(AbiParam::new(ptr_type));

        // Host Memory
        func.signature.params.push(AbiParam::new(ptr_type));

        // Ephemeral Memory
        func.signature.params.push(AbiParam::new(ptr_type));

        // Pc
        func.signature.params.push(AbiParam::new(ptr_type));

        // Exit
        func.signature.params.push(AbiParam::new(ptr_type));

        // Create a builder for the function
        let mut builder = frontend::FunctionBuilder::new(func, ctx);

        // Create the entry block
        let entry_block = builder.create_block();

        // Add the function params to the block
        builder.append_block_params_for_function_params(entry_block);

        // Switch to the entry block
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let registers = builder.block_params(entry_block)[0];
        let host_memory = builder.block_params(entry_block)[1];
        let ephemeral_mem = builder.block_params(entry_block)[2];
        let pc = builder.block_params(entry_block)[3];
        let exit = builder.block_params(entry_block)[4];

        Self {
            builder,
            ptr_type,
            // SAFETY: The following types are always valid.
            int_8_type: unsafe { Type::int(8).unwrap_unchecked() },
            int_16_type: unsafe { Type::int(16).unwrap_unchecked() },
            int_32_type: unsafe { Type::int(32).unwrap_unchecked() },
            int_64_type: unsafe { Type::int(64).unwrap_unchecked() },
            builtins,
            exit_ptr: exit,
            registers_ptr: registers,
            pc_ptr: pc,
            host_memory_ptr: host_memory,
            ephemeral_mem_ptr: ephemeral_mem,
        }
    }

    pub fn constant_32(&mut self, value: impl CastI64) -> Value {
        self.builder.ins().iconst(self.int_32_type, value.as_i64())
    }

    pub fn constant_16(&mut self, value: impl CastI64) -> Value {
        self.builder.ins().iconst(self.int_16_type, value.as_i64())
    }

    pub fn constant_8(&mut self, value: impl CastI64) -> Value {
        self.builder.ins().iconst(self.int_8_type, value.as_i64())
    }
}

impl<'b> BlockBuilder<'b> {
    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic(), pc = pc))]
    pub fn translate_instruction(&mut self, instruction: &Instruction, pc: u32) -> BuilderResult {
        if instruction.is_alu_instruction() {
            self.translate_alu(instruction, pc)
        } else if instruction.is_branch_instruction() {
            self.translate_branch(instruction, pc)
        } else if instruction.is_memory_load_instruction() {
            self.translate_load(instruction, pc)
        } else if instruction.is_memory_store_instruction() {
            self.translate_store(instruction, pc)
        } else if instruction.is_jump_instruction() {
            self.translate_jump(instruction, pc)
        } else if instruction.is_ebreak_instruction() {
            self.translate_ebreak(instruction, pc)
        } else if instruction.is_auipc_instruction() {
            self.translate_auipc(instruction, pc)
        } else {
            unreachable!()
        }
    }

    #[allow(clippy::too_many_lines)]
    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic(), pc = pc))]
    fn translate_alu(&mut self, instruction: &Instruction, pc: u32) -> BuilderResult {
        let (rd, rs1, rs2) = self.alu_rr(instruction);

        match instruction.opcode {
            Opcode::ADD => {
                let result = self.builder.ins().iadd(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::SUB => {
                let result = self.builder.ins().isub(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::XOR => {
                let result = self.builder.ins().bxor(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::OR => {
                let result = self.builder.ins().bor(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::AND => {
                let result = self.builder.ins().band(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::SLL => {
                let result = self.builder.ins().ishl(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::SRL => {
                let result = self.builder.ins().ushr(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::SRA => {
                let result = self.builder.ins().sshr(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::SLT => {
                let result = self.builder.ins().icmp(IntCC::SignedLessThan, rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::SLTU => {
                let result = self.builder.ins().icmp(IntCC::UnsignedLessThan, rs1, rs2);
                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::MUL => {
                let result = self.builder.ins().imul(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::MULH => {
                let result = self.builder.ins().smulhi(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::MULHU => {
                let result = self.builder.ins().umulhi(rs1, rs2);

                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::MULHSU => {
                // Extend rs1 (signed) and rs2 (unsigned) to 64 bits
                let rs1_signed = self.builder.ins().sextend(self.int_64_type, rs1); // Sign-extend rs1 to 64 bits
                let rs2_unsigned = self.builder.ins().uextend(self.int_64_type, rs2); // Zero-extend rs2 to 64 bits

                // The result of this is 128 bits, but we only get the high 64 bits.
                let high_bits = self.builder.ins().smulhi(rs1_signed, rs2_unsigned);

                // Truncate the high 64 bits to 32 bits.
                let result = self.builder.ins().ireduce(self.int_32_type, high_bits);

                // Store the high 32 bits into rd (as a 32-bit value)
                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::DIV | Opcode::DIVU | Opcode::REM | Opcode::REMU => {
                self.translate_branching_alu(instruction.opcode, rd, rs1, rs2);
            }
            _ => unreachable!(),
        }

        BuilderResult::Continue
    }

    fn translate_ecall(&mut self, instruction: &Instruction, pc: u32) -> BuilderResult {
        // Load the syscall id from the registers (5 is the syscall id register)
        let syscall_id = self.builder.ins().load(
            self.int_32_type,
            MemFlags::trusted(),
            self.registers_ptr,
            Register::X5.register_offset(),
        );

        // Load the data registers for the syscall
        let a = self.builder.ins().load(
            self.int_32_type,
            MemFlags::trusted(),
            self.registers_ptr,
            Register::X10.register_offset(),
        );

        let b = self.builder.ins().load(
            self.int_32_type,
            MemFlags::trusted(),
            self.registers_ptr,
            Register::X11.register_offset(),
        );

        // Call the syscall
        self.handle_syscall(pc, syscall_id, a, b);

        BuilderResult::Ecall
    }

    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic(), pc = pc))]
    fn translate_branch(&mut self, instruction: &Instruction, pc: u32) -> BuilderResult {
        let (rs1, rs2, imm) = instruction.b_type();

        let rs1_val = self.builder.ins().load(
            self.int_32_type,
            MemFlags::trusted(),
            self.registers_ptr,
            rs1.register_offset(),
        );

        let rs2_val = self.builder.ins().load(
            self.int_32_type,
            MemFlags::trusted(),
            self.registers_ptr,
            rs2.register_offset(),
        );

        let cond = match instruction.opcode {
            Opcode::BEQ => self.builder.ins().icmp(IntCC::Equal, rs1_val, rs2_val),
            Opcode::BNE => self.builder.ins().icmp(IntCC::NotEqual, rs1_val, rs2_val),
            Opcode::BLT => self.builder.ins().icmp(IntCC::SignedLessThan, rs1_val, rs2_val),
            Opcode::BGE => {
                self.builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, rs1_val, rs2_val)
            }
            Opcode::BLTU => self.builder.ins().icmp(IntCC::UnsignedLessThan, rs1_val, rs2_val),
            Opcode::BGEU => {
                self.builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, rs1_val, rs2_val)
            }
            _ => unreachable!(),
        };

        let branched = self.builder.create_block();
        let not_branched = self.builder.create_block();

        self.builder.ins().brif(cond, branched, &[], not_branched, &[]);

        BuilderResult::Branch { target: pc + imm, branched, not_branched }
    }

    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic(), pc = pc))]
    fn translate_load(&mut self, instruction: &Instruction, pc: u32) -> BuilderResult {
        let (rd, addr) = self.load_rr(instruction);
        let word = self.call_load(addr);

        match instruction.opcode {
            Opcode::LW => {
                // Store the result into the rd register
                self.builder.ins().store(
                    MemFlags::trusted(),
                    word,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::LB => {
                // Load only the lower byte of the word
            }
            _ => unreachable!(),
        }

        BuilderResult::Continue
    }

    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic(), pc = pc))]
    fn translate_store(&mut self, instruction: &Instruction, pc: u32) -> BuilderResult {
        todo!()
    }

    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic(), pc = pc))]
    fn translate_jump(&mut self, instruction: &Instruction, pc: u32) -> BuilderResult {
        debug_assert!(instruction.is_jump_instruction());

        match instruction.opcode {
            Opcode::JALR => {
                // rd ← pc + 4, pc ← (rs1 + imm) & ∼1
                let inc = pc + 4;
                let pc_plus_4 = self.constant_32(inc);

                let rd = instruction.op_a;
                let rs1 = instruction.op_b;
                let imm = self.constant_32(instruction.op_c);

                // Store the pc + 4 into the rd register.
                // rd <- pc + 4
                self.builder.ins().store(MemFlags::trusted(), pc_plus_4, self.registers_ptr, rd);

                // pc <- (rs1 + imm)
                let v_rs1 = self.builder.ins().load(
                    self.int_32_type,
                    MemFlags::trusted(),
                    self.registers_ptr,
                    rs1.register_offset(),
                );

                let rs1_plus_imm = self.builder.ins().iadd(v_rs1, imm);

                // Store the jump target.
                // This is the end of this function. So set the pc in the host to the jump target.
                self.set_pc(rs1_plus_imm);

                // Exit the function
                self.builder.ins().return_(&[]);

                BuilderResult::Jalr
            }
            Opcode::JAL => {
                // rd ← pc + 4, pc ← pc + imm
                let inc = pc + 4;
                let pc_plus_4 = self.constant_32(inc);

                let rd = instruction.op_a;

                // rd <- pc + 4
                self.builder.ins().store(
                    MemFlags::trusted(),
                    pc_plus_4,
                    self.registers_ptr,
                    rd.register_offset(),
                );

                // We dont need to update the `self.pc_ptr` here because any jumps would be in the jit function

                BuilderResult::Jal(pc + instruction.op_b)
            }
            _ => unreachable!(),
        }
    }

    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic(), pc = pc))]
    fn translate_ebreak(&mut self, instruction: &Instruction, pc: u32) -> BuilderResult {
        todo!()
    }

    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic(), pc = pc))]
    fn translate_auipc(&mut self, instruction: &Instruction, pc: u32) -> BuilderResult {
        debug_assert!(instruction.opcode == Opcode::AUIPC);

        let (rd, imm) = instruction.u_type();

        let pc_plus_imm = pc + imm;
        let pc_plus_imm = self.constant_32(pc_plus_imm);

        // rd <- pc + imm
        self.builder.ins().store(
            MemFlags::trusted(),
            pc_plus_imm,
            self.registers_ptr,
            rd.register_offset(),
        );

        BuilderResult::Auipc
    }

    /// Translate a branching ALU instruction that checks for 0 divisors.
    #[tracing::instrument(skip_all, fields(opcode = op.mnemonic()))]
    fn translate_branching_alu(&mut self, op: Opcode, rd: u8, rs1: Value, rs2: Value) {
        debug_assert!(
            op == Opcode::DIV || op == Opcode::DIVU || op == Opcode::REM || op == Opcode::REMU
        );

        let zero_branch = self.builder.create_block();
        let not_zero_branch = self.builder.create_block();
        let merge_branch = self.builder.create_block();

        let rs1_param = self.builder.append_block_param(not_zero_branch, self.int_32_type);
        let rs2_param = self.builder.append_block_param(not_zero_branch, self.int_32_type);

        let zero = self.constant_32(0_u32);
        let rs2_is_zero = self.builder.ins().icmp(IntCC::Equal, rs2, zero);

        self.builder.ins().brif(rs2_is_zero, zero_branch, &[], not_zero_branch, &[rs1, rs2]);

        self.builder.switch_to_block(zero_branch);
        self.builder.seal_block(zero_branch);

        // Were in the zero block, so lets set the result to 0
        let zero = self.constant_32(0_u32);
        self.builder.ins().store(
            MemFlags::trusted(),
            zero,
            self.registers_ptr,
            rd.register_offset(),
        );
        self.builder.ins().jump(merge_branch, &[]);

        self.builder.switch_to_block(not_zero_branch);
        self.builder.seal_block(not_zero_branch);

        match op {
            Opcode::DIV => {
                let result = self.builder.ins().sdiv(rs1_param, rs2_param);
                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::DIVU => {
                let result = self.builder.ins().udiv(rs1_param, rs2_param);
                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::REM => {
                let result = self.builder.ins().srem(rs1_param, rs2_param);
                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            Opcode::REMU => {
                let result = self.builder.ins().urem(rs1_param, rs2_param);
                self.builder.ins().store(
                    MemFlags::trusted(),
                    result,
                    self.registers_ptr,
                    rd.register_offset(),
                );
            }
            _ => unreachable!(),
        }

        self.builder.ins().jump(merge_branch, &[]);
        self.builder.switch_to_block(merge_branch);
        self.builder.seal_block(merge_branch);
    }
}

impl<'a> BlockBuilder<'a> {
    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic()))]
    fn alu_rr(&mut self, instruction: &Instruction) -> (u8, Value, Value) {
        debug_assert!(instruction.is_alu_instruction());

        if !instruction.imm_c {
            let (rd, rs1, rs2) = (instruction.op_a, instruction.op_b, instruction.op_c);

            tracing::trace!("loading rs1 = {rs1} and rs2 =  {rs2} values");
            tracing::trace!("rd = {rd}");

            // load the rs1 and rs2 values
            let v_rs1 = self.builder.ins().load(
                self.int_32_type,
                MemFlags::trusted(),
                self.registers_ptr,
                rs1.register_offset(),
            );

            let v_rs2 = self.builder.ins().load(
                self.int_32_type,
                MemFlags::trusted(),
                self.registers_ptr,
                rs2.register_offset(),
            );

            (rd, v_rs1, v_rs2)
        } else if !instruction.imm_b && instruction.imm_c {
            let (rd, rs1, imm) = (instruction.op_a, instruction.op_b, instruction.op_c);

            tracing::trace!("loading rs1 = {rs1} and have imm c = {imm} value");
            tracing::trace!("rd = {rd}");

            // Adding an immediate to a register value.
            let v_rs1 = self.builder.ins().load(
                self.int_32_type,
                MemFlags::trusted(),
                self.registers_ptr,
                rs1.register_offset(),
            );

            let v_imm = self.constant_32(imm);

            (rd, v_rs1, v_imm)
        } else {
            debug_assert!(instruction.imm_b && instruction.imm_c);
            // Adding two immediates.
            let (rd, imm_1, imm_2) = (instruction.op_a, instruction.op_b, instruction.op_c);

            tracing::trace!("have imm_1 = {imm_1} and imm_2 = {imm_2} values");
            tracing::trace!("rd = {rd}");

            let v_imm_1 = self.constant_32(imm_1);
            let v_imm_2 = self.constant_32(imm_2);

            (rd, v_imm_1, v_imm_2)
        }
    }

    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic()))]
    fn load_rr(&mut self, instruction: &Instruction) -> (Register, Value) {
        debug_assert!(instruction.is_memory_load_instruction());

        let (rd, rs1, imm) = instruction.i_type();

        let v_rs1 = self.builder.ins().load(
            self.int_32_type,
            MemFlags::trusted(),
            self.registers_ptr,
            rs1.register_offset(),
        );

        let addr = self.builder.ins().iadd_imm(v_rs1, imm as i64);

        (rd, self.align_to_word(addr))
    }

    #[tracing::instrument(skip_all, fields(opcode = instruction.opcode.mnemonic()))]
    fn store_rr(&mut self, instruction: &Instruction) -> (Value, Value) {
        debug_assert!(instruction.is_memory_store_instruction());

        // m(rs1 + imm) <- rs2
        let (rs1, rs2, imm) = instruction.s_type();

        let v_rs1 = self.builder.ins().load(
            self.int_32_type,
            MemFlags::trusted(),
            self.registers_ptr,
            rs1.register_offset(),
        );

        let v_word = self.builder.ins().load(
            self.int_32_type,
            MemFlags::trusted(),
            self.registers_ptr,
            rs2.register_offset(),
        );

        let addr = self.builder.ins().iadd_imm(v_rs1, imm as i64);
        let v_addr = self.align_to_word(addr);

        (v_addr, v_word)
    }
}

impl<'a> BlockBuilder<'a> {
    #[tracing::instrument(skip_all, fields(pc = pc))]
    fn handle_syscall(&mut self, pc: u32, syscall_id: Value, a: Value, b: Value) {
        // todo(n)
        // This (incorrectly) assumes were given the the exit syscall..
        self.exit_unconstrained(pc);

        // todo: handle syscall write
    }

    pub(crate) fn set_pc(&mut self, pc: Value) {
        self.builder.ins().store(MemFlags::trusted(), pc, self.pc_ptr, 0);
    }

    #[tracing::instrument(skip(self))]
    pub(crate) fn exit_unconstrained(&mut self, pc: u32) {
        let one = self.constant_32(1_u32);

        // Set the exit flag to 1
        self.builder.ins().store(MemFlags::trusted(), one, self.exit_ptr, 0);

        let pc = pc + 4;
        let pc_plus_four = self.constant_32(pc);

        // This handles `EXIT_UNCONSTRAINED`
        self.set_pc(pc_plus_four);

        // Also return from the function
        self.builder.ins().return_(&[]);
    }
}

impl<'a> BlockBuilder<'a> {
    fn get_builtin(&self, builtin: Builtin) -> FuncRef {
        debug_assert!(self.builtins.contains_key(&builtin));

        unsafe { *self.builtins.get(&builtin).unwrap_unchecked() }
    }

    // Attemps to call a builtin function.
    // Returns the result of the builtin function.
    // Assumes that the params are valid.
    fn call_builtin(&mut self, params: &[Value], builtin: Builtin) -> &[Value] {
        let func = self.get_builtin(builtin);

        let result = self.builder.ins().call(func, params);

        self.builder.inst_results(result)
    }

    /// Loads a word from memory
    /// Assumes that the address is aligned to the word.
    fn call_load(&mut self, addr: Value) -> Value {
        let params = vec![self.host_memory_ptr, self.ephemeral_mem_ptr, addr];

        self.call_builtin(&params, Builtin::Load)
            .first()
            .copied()
            .expect("failed to get load result")
    }

    /// Stores a word into the ephemeral memory.
    fn call_store(&mut self, addr: Value, data: Value) {
        let params = vec![self.ephemeral_mem_ptr, addr, data];

        self.call_builtin(&params, Builtin::Store);
    }

    #[inline]
    fn align_to_word(&mut self, addr: Value) -> Value {
        let residue = self.builder.ins().urem_imm(addr, 4);
        let addr = self.builder.ins().isub(addr, residue);

        addr
    }
}

pub enum BuilderResult {
    Branch {
        // The pc we would go to if we branched
        target: u32,
        // The block we would go to if we branched
        branched: Block,
        // The block we would go to if we did not branch
        not_branched: Block,
    },
    Ecall,
    Continue,
    Auipc,
    Jal(u32),
    Jalr,
}

use helper_traits::{CastI64, RegisterIndex};

mod helper_traits {
    pub(super) trait CastI64 {
        fn as_i64(self) -> i64;
    }

    pub(super) trait RegisterIndex {
        fn register_offset(self) -> i32;
    }

    macro_rules! impl_register_index {
        ($($t:ty),*) => {
            $(impl RegisterIndex for $t {
                #[inline(always)]
                /// The offset of the register in the registers array. (in bytes)
                fn register_offset(self) -> i32 {
                    ((self as u32) * size_of::<u32>() as u32) as i32
                }
            })*
        };
    }

    impl_register_index!(u8, u16, u32, super::Register);

    macro_rules! impl_cast_i64 {
        ($($t:ty),*) => {
            $(impl CastI64 for $t {
                #[inline(always)]
                fn as_i64(self) -> i64 {
                    self as i64
                }
            })*
        };
    }

    impl_cast_i64!(u8, u16, u32, u64);
}
