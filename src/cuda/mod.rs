pub mod ast;
mod codegen;
mod pprint;

use ast::*;
use crate::ir::ast as ir_ast;
use crate::gpu;
use crate::utils::debug::*;
use crate::utils::err::*;

pub fn codegen(ir_ast: ir_ast::Ast, debug_env: &DebugEnv) -> CompileResult<Ast> {
    // Convert the IR AST to a general GPU AST.
    let gpu_ast = gpu::from_general_ir(ir_ast, debug_env)?;

    // Convert the GPU AST to a CUDA C++ AST.
    codegen::from_gpu_ir(gpu_ast)
}
