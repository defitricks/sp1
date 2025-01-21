use crate::{
    events::MemoryRecord,
    jit::{Engine, VmContext},
    memory::{Memory, PagedMemory},
    Instruction, Opcode, Program,
};

use std::sync::Once;

static TRACING: Once = Once::new();

macro_rules! init_tracing {
    () => {
        TRACING.call_once(|| {
            tracing_subscriber::fmt::init();
        });
    };
    ($span_name:expr) => {
        TRACING.call_once(|| {
            tracing_subscriber::fmt::init();
        });

        let __span = tracing::span!(tracing::Level::INFO, $span_name);
        let __span = __span.enter();
    };
}

/// Run the program without memory access.
///
/// Note: Trying to get mem will seg fault dont do it
fn run_program_no_mem(engine: &mut Engine, program: &Program, registers: *mut u32, pc: u32) -> u32 {
    run_program(
        engine,
        program,
        &mut VmContext::new(registers, std::ptr::null_mut(), std::ptr::null_mut(), pc),
    )
}

fn run_program_with_mem(
    engine: &mut Engine,
    program: &Program,
    registers: *mut u32,
    pc: u32,
    memory: *mut Memory<MemoryRecord>,
) -> (Memory<MemoryRecord>, u32) {
    let mut eph = Memory::new_preallocated();
    let eph_ptr = &mut eph as *mut Memory<MemoryRecord>;

    let pc_end = run_program(engine, program, &mut VmContext::new(registers, memory, eph_ptr, pc));

    (eph, pc_end)
}

/// Run the program without memory access.
///
/// Note: Trying to get mem will seg fault dont do it
#[allow(clippy::needless_pass_by_value)]
fn run_program(engine: &mut Engine, program: &Program, context: &mut VmContext) -> u32 {
    // Start executing the program
    // Our current JIT strategy only compiles up until the next `JALR` instruction.
    loop {
        if let Some(entry_point) = engine.entry_points.get(&context.pc) {
            unsafe {
                entry_point.call(context);
            }

            // We are exiting unconstrained mode.
            if context.exit == 1 {
                break context.pc;
            }
        } else {
            engine.compile_and_link(program, context.pc).expect("failed to compile");
            continue;
        }
    }
}

mod builtins {
    use super::*;

    #[test]
    fn test_load() {
        init_tracing!("test_load");

        // Get address from register 2, and store it in register 1
        let program = Program::new(vec![Instruction::new(Opcode::LW, 1, 2, 0, false, true)], 0, 0);

        let mut engine = Engine::new(0);
        engine.add_exit();

        let mut registers = [0; 32];
        registers[2] = 400;

        let mut memory = Memory::new_preallocated();
        memory.insert(400, MemoryRecord { value: 100, ..Default::default() });

        run_program_with_mem(&mut engine, &program, registers.as_mut_ptr(), 0, &mut memory);

        // Assert weve read address 400 into register 1
        assert_eq!(registers[1], 100);
    }
}

mod control_flow {
    use super::*;

    #[test]
    fn test_jal() {
        init_tracing!("test_jal");

        let program =
            Program::new(vec![Instruction::new(Opcode::JAL, 1, 100, 0, false, false)], 0, 0);

        let mut engine = Engine::new(0);
        engine.add_exit();

        let mut registers = [0; 32];
        let pc = run_program_no_mem(&mut engine, &program, registers.as_mut_ptr(), 0);

        // check rd <- pc + 4
        assert_eq!(registers[1], 4);

        // At the end of the program, the test harness will have inserted a mock "exit" syscall
        // so expect the last pc to be 104
        assert_eq!(pc, 104);
    }

    #[test]
    fn test_beq() {
        init_tracing!("test_beq");
    }
}

mod alu {
    use super::*;

    #[test]
    fn test_add_rr() {
        init_tracing!("test_add_rr");

        // X3 <- X1 + X2
        let program =
            Program::new(vec![Instruction::new(Opcode::ADD, 3, 1, 2, false, false)], 0, 0);

        let mut engine = Engine::new(0);
        engine.add_exit();

        // Set the values of the registers
        let mut registers = [0; 32];
        {
            registers[1] = 5;
            registers[2] = 10;
        }
        let rref = registers.as_mut_ptr();

        run_program_no_mem(&mut engine, &program, rref, 0);

        assert_eq!(registers[3], 15);
    }

    #[test]
    fn test_add_1_imm() {
        init_tracing!("test_add_1_imm");

        // X3 <- X1 + imm(1)
        let program = Program::new(vec![Instruction::new(Opcode::ADD, 3, 1, 1, false, true)], 0, 0);

        let mut engine = Engine::new(0);
        engine.add_exit();

        let mut registers = [0; 32];
        registers[1] = 5;

        run_program_no_mem(&mut engine, &program, registers.as_mut_ptr(), 0);

        assert_eq!(registers[3], 6);
    }

    #[test]
    fn test_add_2_imm() {
        init_tracing!("test_add_2_imm");

        // X10 <- imm(1) + imm(2)
        let program = Program::new(vec![Instruction::new(Opcode::ADD, 10, 1, 2, true, true)], 0, 0);

        let mut engine = Engine::new(0);
        engine.add_exit();

        let mut registers = [0; 32];

        run_program_no_mem(&mut engine, &program, registers.as_mut_ptr(), 0);

        assert_eq!(registers[10], 3);
    }

    #[test]
    fn test_div_rr() {
        init_tracing!("test_div_rr");

        let program =
            Program::new(vec![Instruction::new(Opcode::DIV, 3, 1, 2, false, false)], 0, 0);

        let mut engine = Engine::new(0);
        engine.add_exit();

        let mut registers = [0; 32];
        registers[1] = 10;
        registers[2] = 5;

        run_program_no_mem(&mut engine, &program, registers.as_mut_ptr(), 0);

        assert_eq!(registers[3], 2);
    }

    #[test]
    fn test_mul_rr() {
        init_tracing!("test_mul_rr");

        let program =
            Program::new(vec![Instruction::new(Opcode::MUL, 3, 1, 2, false, false)], 0, 0);

        let mut engine = Engine::new(0);
        engine.add_exit();
        let mut registers = [0; 32];
        registers[1] = 5;
        registers[2] = 10;

        run_program_no_mem(&mut engine, &program, registers.as_mut_ptr(), 0);

        assert_eq!(registers[3], 50);
    }

    #[test]
    fn test_and_rr() {
        init_tracing!("test_and_rr");

        let program =
            Program::new(vec![Instruction::new(Opcode::AND, 3, 1, 2, false, false)], 0, 0);

        let mut engine = Engine::new(0);
        engine.add_exit();

        let mut registers = [0; 32];
        registers[1] = 5;
        registers[2] = 11;

        run_program_no_mem(&mut engine, &program, registers.as_mut_ptr(), 0);

        assert_eq!(registers[3], 5 & 11);
    }

    #[test]
    fn test_xor_rr() {
        init_tracing!("test_xor_rr");

        let program =
            Program::new(vec![Instruction::new(Opcode::XOR, 3, 1, 2, false, false)], 0, 0);

        let mut engine = Engine::new(0);
        engine.add_exit();

        let mut registers = [0; 32];
        registers[1] = 5;
        registers[2] = 11;

        run_program_no_mem(&mut engine, &program, registers.as_mut_ptr(), 0);

        assert_eq!(registers[3], 5 ^ 11);
    }

    #[test]
    fn test_or_rr() {
        init_tracing!("test_or_rr");

        let program = Program::new(vec![Instruction::new(Opcode::OR, 3, 1, 2, false, false)], 0, 0);

        let mut engine = Engine::new(0);
        engine.add_exit();

        let mut registers = [0; 32];
        registers[1] = 5;
        registers[2] = 11;

        run_program_no_mem(&mut engine, &program, registers.as_mut_ptr(), 0);

        assert_eq!(registers[3], 5 | 11);
    }
}
