use crate::parir_compile_error;
use crate::ir::ast as ir_ast;
use crate::py::ast::{BinOp, Builtin, ElemSize};
use crate::py::ast as py_ast;
use crate::gpu::ast as gpu_ast;
use crate::cuda::ast as cu_ast;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::pprint::PrettyPrint;

// Trait allowing us to construct an expression given a literal floating-point value and an element
// size determining its type.
pub trait ExprLit {
    fn generate_literal(v: f64, sz: &ElemSize, i: Info) -> Self where Self: Sized;

    fn to_int_lit(v: f64) -> i64 {
        v as i64
    }
    fn to_bool_lit(v: f64) -> bool {
        v != 0.0
    }
}

impl ExprLit for py_ast::Expr {
    fn generate_literal(v: f64, sz: &ElemSize, i: Info) -> py_ast::Expr {
        let ty = py_ast::Type::Tensor {sz: sz.clone(), shape: vec![]};
        match sz {
            ElemSize::Bool => {
                py_ast::Expr::Bool {v: py_ast::Expr::to_bool_lit(v), ty, i}
            },
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => {
                py_ast::Expr::Int {v: py_ast::Expr::to_int_lit(v), ty, i}
            },
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => {
                py_ast::Expr::Float {v, ty, i}
            }
        }
    }
}

impl ExprLit for ir_ast::Expr {
    fn generate_literal(v: f64, sz: &ElemSize, i: Info) -> ir_ast::Expr {
        let ty = ir_ast::Type::Tensor {sz: sz.clone(), shape: vec![]};
        match sz {
            ElemSize::Bool => {
                ir_ast::Expr::Bool {v: ir_ast::Expr::to_bool_lit(v), ty, i}
            },
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => {
                ir_ast::Expr::Int {v: ir_ast::Expr::to_int_lit(v), ty, i}
            },
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => {
                ir_ast::Expr::Float {v, ty, i}
            }
        }
    }
}

impl ExprLit for gpu_ast::Expr {
    fn generate_literal(v: f64, sz: &ElemSize, i: Info) -> gpu_ast::Expr {
        let ty = gpu_ast::Type::Scalar {sz: sz.clone()};
        match sz {
            ElemSize::Bool => {
                gpu_ast::Expr::Bool {v: gpu_ast::Expr::to_bool_lit(v), ty, i}
            },
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => {
                gpu_ast::Expr::Int {v: gpu_ast::Expr::to_int_lit(v), ty, i}
            },
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => {
                gpu_ast::Expr::Float {v, ty, i}
            }
        }
    }
}

impl ExprLit for cu_ast::Expr {
    fn generate_literal(v: f64, sz: &ElemSize, i: Info) -> cu_ast::Expr {
        let ty = cu_ast::Type::Scalar {sz: sz.clone()};
        match sz {
            ElemSize::Bool => {
                cu_ast::Expr::Bool {v: cu_ast::Expr::to_bool_lit(v), ty, i}
            },
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => {
                cu_ast::Expr::Int {v: cu_ast::Expr::to_int_lit(v), ty, i}
            },
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => {
                cu_ast::Expr::Float {v, ty, i}
            }
        }
    }
}

pub fn neutral_element<T: ExprLit>(
    op: &BinOp,
    sz: &ElemSize,
    i: &Info
) -> Option<T> {
    let i = i.clone();
    match op {
        BinOp::Add => Some(T::generate_literal(0.0, sz, i)),
        BinOp::Mul => Some(T::generate_literal(1.0, sz, i)),
        BinOp::Max => Some(T::generate_literal(f64::NEG_INFINITY, sz, i)),
        BinOp::Min => Some(T::generate_literal(f64::INFINITY, sz, i)),
        _ => None
    }
}

pub fn builtin_to_reduction_op(func: &Builtin) -> Option<BinOp> {
    match func {
        Builtin::Sum => Some(BinOp::Add),
        Builtin::Prod => Some(BinOp::Mul),
        Builtin::Max => Some(BinOp::Max),
        Builtin::Min => Some(BinOp::Min),
        _ => None
    }
}

/// The 'and' and 'or' operators are short-circuiting in both C and Python. However, we do not want
/// to generate short-circuiting code, as the warp-level reductions will get stuck unless all
/// threads have the same value (because some threads short-circuit, thereby ignoring the warp sync
/// intrinsic call). To work around this, we use the bitwise operations, which are equivalent for
/// boolean values assuming they are encoded as 0 or 1 (not necessarily true in C).
fn non_short_circuiting_op(op: ir_ast::BinOp) -> ir_ast::BinOp {
    match op {
        BinOp::And => BinOp::BitAnd,
        BinOp::Or => BinOp::BitOr,
        _ => op
    }
}

fn extract_bin_op(
    expr: ir_ast::Expr
) -> CompileResult<(ir_ast::Expr, ir_ast::BinOp, ir_ast::Expr, ElemSize, Info)> {
    let i = expr.get_info();
    match expr {
        ir_ast::Expr::BinOp {lhs, op, rhs, ty, i} => {
            match ty {
                ir_ast::Type::Tensor {sz, shape} if shape.is_empty() => {
                    Ok((*lhs, non_short_circuiting_op(op), *rhs, sz, i))
                },
                _ => parir_compile_error!(i, "Expected the result of reduction \
                                              to be a scalar value, found {0}",
                                              ty.pprint_default())
            }
        },
        ir_ast::Expr::Convert {e, ..} => extract_bin_op(*e),
        _ => {
            parir_compile_error!(i, "RHS of reduction statement should be a \
                                     binary operation.")
        }
    }
}

fn unwrap_convert(e: &ir_ast::Expr) -> ir_ast::Expr {
    match e {
        ir_ast::Expr::Convert {e, ..} => unwrap_convert(e),
        _ => e.clone()
    }
}

pub fn extract_reduction_operands(
    mut body: Vec<ir_ast::Stmt>,
    i: &Info
) -> CompileResult<(ir_ast::Expr, ir_ast::BinOp, ir_ast::Expr, ElemSize, Info)> {
    // The reduction loop body must contain a single statement.
    if body.len() == 1 {
        // The single statement must be a single (re)assignment.
        if let ir_ast::Stmt::Assign {dst, expr, ..} = body.remove(0) {
            // The right-hand side should be a binary operation, so we extract its constituents.
            let (lhs, op, rhs, sz, i) = extract_bin_op(expr)?;
            // The destination of the assignment must either be a variable or a tensor access.
            match dst {
                ir_ast::Expr::Var {..} | ir_ast::Expr::TensorAccess {..} => {
                    // The assignment destination must be equal to the left-hand side of the
                    // reduction operation.
                    if dst == unwrap_convert(&lhs) {
                        Ok((dst, op, rhs, sz, i))
                    } else {
                        let msg = format!(
                            "Invalid reduction. Left-hand side of binary \
                             operation {0} is not equal to the assignment \
                             target {1}.",
                             lhs.pprint_default(), dst.pprint_default()
                        );
                        parir_compile_error!(i, "{}", msg)
                    }
                },
                _ => {
                    parir_compile_error!(i, "Left-hand side of reduction must \
                                             be a variable or tensor access.")
                }
            }
        } else {
            parir_compile_error!(i, "Reduction for-loop statement must be an \
                                     assignment.")
        }
    } else {
        parir_compile_error!(i, "Reduction for-loop must contain a single \
                                 statement.")
    }
}
