/// This file contains an analysis finding places where many threads within a block write to the
/// same global memory address. If all threads of a block perform a write to the same global memory
/// location, this may produce an invalid result. For instance, the tests in test/test_syrk.py
/// reproduces this problem in CUDA running on an A100 when using 512 or 1024 threads (when this
/// transformation is not included).
///
/// The analysis is performed in two steps. First, we collect a set of all variables whose value
/// depends on the thread index. This comes either from the variable in the innermost for-loop or
/// any expressions computed based on it. Second, we transform global writes to memory where the
/// index is independent of the thread index, so that only the first thread performs the write.

use super::ast::*;
use crate::parir_compile_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::BTreeSet;

fn thread_index_dependent_expr_helper(
    acc: bool,
    vars: &BTreeSet<Name>,
    expr: &Expr
) -> bool {
    if acc {
        acc
    } else {
        match expr {
            Expr::Var {id, ..} if vars.contains(&id) => true,
            Expr::ThreadIdx {..} => true,
            _ => expr.sfold(acc, |acc, e| thread_index_dependent_expr_helper(acc, vars, e))
        }
    }
}

fn thread_index_dependent_expr(
    vars: &BTreeSet<Name>,
    expr: &Expr
) -> bool {
    thread_index_dependent_expr_helper(false, vars, expr)
}

fn extract_assign_target_id(e: &Expr) -> CompileResult<Name> {
    match e {
        Expr::Var {id, ..} => Ok(id.clone()),
        Expr::ArrayAccess {target, ..} => extract_assign_target_id(&target),
        _ => {
            parir_compile_error!(e.get_info(), "Unexpected target of assignment")
        }
    }
}

fn find_thread_index_dependent_variables_stmt(
    acc: CompileResult<BTreeSet<Name>>,
    stmt: &Stmt
) -> CompileResult<BTreeSet<Name>> {
    let mut acc = acc?;
    match stmt {
        Stmt::Definition {id, expr, ..} if thread_index_dependent_expr(&acc, expr) => {
            acc.insert(id.clone());
            Ok(acc)
        },
        Stmt::Assign {dst, expr, ..} if thread_index_dependent_expr(&acc, expr) => {
            let target_id = extract_assign_target_id(dst)?;
            acc.insert(target_id);
            Ok(acc)
        },
        Stmt::For {var, init, body, ..} => {
            if thread_index_dependent_expr(&acc, init) {
                acc.insert(var.clone());
            }
            body.sfold(Ok(acc), find_thread_index_dependent_variables_stmt)
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::While {..} |
        Stmt::If {..} | Stmt::Scope {..} | Stmt::SynchronizeBlock {..} |
        Stmt::KernelLaunch {..} | Stmt::Alloc {..} | Stmt::Dealloc {..} => {
            stmt.sfold(Ok(acc), find_thread_index_dependent_variables_stmt)
        }
    }
}

fn find_thread_index_dependent_variables_top(
    acc: CompileResult<BTreeSet<Name>>,
    top: &Top
) -> CompileResult<BTreeSet<Name>> {
    match top {
        Top::DeviceFunDef {body, ..} => {
            body.sfold(acc, find_thread_index_dependent_variables_stmt)
        },
        Top::HostFunDef {..} | Top::StructDef {..} => acc
    }
}

fn find_thread_index_dependent_variables(ast: &Ast) -> CompileResult<BTreeSet<Name>> {
    ast.iter()
        .fold(Ok(BTreeSet::new()), find_thread_index_dependent_variables_top)
}

fn transform_thread_independent_memory_writes_stmt(
    mut acc: Vec<Stmt>,
    stmt: Stmt,
    vars: &BTreeSet<Name>
) -> Vec<Stmt> {
    let i = stmt.get_info();
    match stmt {
        Stmt::Assign {dst: Expr::ArrayAccess {ref idx, i: ref ii, ..}, ..} => {
            if thread_index_dependent_expr(&vars, idx.as_ref()) {
                acc.push(stmt);
            } else {
                let i64_ty = Type::Scalar {sz: ElemSize::I64};
                acc.push(Stmt::If {
                    cond: Expr::BinOp {
                        lhs: Box::new(Expr::ThreadIdx {
                            dim: Dim::X, ty: i64_ty.clone(), i: ii.clone()
                        }),
                        op: BinOp::Eq,
                        rhs: Box::new(Expr::Int {
                            v: 0, ty: i64_ty.clone(), i: ii.clone()
                        }),
                        ty: Type::Boolean, i: ii.clone()
                    },
                    thn: vec![stmt],
                    els: vec![],
                    i: i.clone()
                });
                acc.push(Stmt::SynchronizeBlock {i});
            }
        },
        Stmt::For {var_ty, var, init, cond, incr, body, i} => {
            let body = transform_thread_independent_memory_writes_stmts(body, &vars);
            acc.push(Stmt::For {var_ty, var, init, cond, incr, body, i});
        },
        Stmt::If {cond, thn, els, i} => {
            let thn = transform_thread_independent_memory_writes_stmts(thn, &vars);
            let els = transform_thread_independent_memory_writes_stmts(els, &vars);
            acc.push(Stmt::If {cond, thn, els, i});
        },
        Stmt::While {cond, body, i} => {
            let body = transform_thread_independent_memory_writes_stmts(body, &vars);
            acc.push(Stmt::While {cond, body, i});
        },
        Stmt::Scope {body, i} => {
            let body = transform_thread_independent_memory_writes_stmts(body, &vars);
            acc.push(Stmt::Scope {body, i});
        },
        _ => acc.push(stmt),
    };
    acc
}

fn transform_thread_independent_memory_writes_stmts(
    stmts: Vec<Stmt>,
    vars: &BTreeSet<Name>
) -> Vec<Stmt> {
    stmts.into_iter()
        .fold(vec![], |acc, s| {
            transform_thread_independent_memory_writes_stmt(acc, s, &vars)
        })
}

fn transform_thread_independent_memory_writes_top(
    top: Top,
    vars: &BTreeSet<Name>
) -> Top {
    match top {
        Top::DeviceFunDef {threads, id, params, body} => {
            let body = transform_thread_independent_memory_writes_stmts(body, vars);
            Top::DeviceFunDef {threads, id, params, body}
        },
        Top::HostFunDef {..} | Top::StructDef {..} => top,
    }
}

fn transform_thread_independent_memory_writes(
    ast: Ast,
    vars: &BTreeSet<Name>
) -> Ast {
    ast.into_iter()
        .map(|top| {
            transform_thread_independent_memory_writes_top(top, vars)
        })
        .collect::<Ast>()
}


pub fn eliminate_block_wide_memory_writes(ast: Ast) -> CompileResult<Ast> {
    let thread_vars = find_thread_index_dependent_variables(&ast)?;
    Ok(transform_thread_independent_memory_writes(ast, &thread_vars))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::info::Info;

    fn for_loop(var: Name, dim: Dim, lo: i64, hi: i64, body: Vec<Stmt>, block: bool) -> Stmt {
        let i64_ty = Type::Scalar {sz: ElemSize::I64};
        let i = || Info::default();
        let rhs = if block {
            Expr::BlockIdx {dim, ty: i64_ty.clone(), i: i()}
        } else {
            Expr::ThreadIdx {dim, ty: i64_ty.clone(), i: i()}
        };
        let init = Expr::BinOp {
            lhs: Box::new(Expr::Int {v: lo, ty: i64_ty.clone(), i: i()}),
            op: BinOp::Add,
            rhs: Box::new(rhs),
            ty: i64_ty.clone(), i: i()
        };
        let cond = Expr::BinOp {
            lhs: Box::new(Expr::Var {id: var.clone(), ty: i64_ty.clone(), i: i()}),
            op: BinOp::Lt,
            rhs: Box::new(Expr::Int {v: hi, ty: i64_ty.clone(), i: i()}),
            ty: i64_ty.clone(), i: i()
        };
        let incr = Expr::BinOp {
            lhs: Box::new(Expr::Var {id: var.clone(), ty: i64_ty.clone(), i: i()}),
            op: BinOp::Add,
            rhs: Box::new(Expr::Int {v: 1, ty: i64_ty.clone(), i: i()}),
            ty: i64_ty.clone(), i: i()
        };
        Stmt::For {
            var_ty: Type::Scalar {sz: ElemSize::I64},
            var, init, cond, incr, body, i: i()
        }
    }

    fn block_dependent_loop(var: Name, dim: Dim, lo: i64, hi: i64, body: Vec<Stmt>) -> Stmt {
        for_loop(var, dim, lo, hi, body, true)
    }

    fn thread_dependent_loop(var: Name, dim: Dim, lo: i64, hi: i64, body: Vec<Stmt>) -> Stmt {
        for_loop(var, dim, lo, hi, body, false)
    }

    fn i() -> Info {
        Info::default()
    }

    fn i64_ty() -> Type {
        Type::Scalar {sz: ElemSize::I64}
    }

    fn id(s: &str) -> Name {
        Name::sym_str(s)
    }

    fn var(id: Name) -> Expr {
        Expr::Var {id, ty: i64_ty(), i: i()}
    }

    fn int(v: i64) -> Expr {
        Expr::Int {v, ty: i64_ty(), i: i()}
    }

    fn tcheck(stmt: Stmt) -> Stmt {
        Stmt::If {
            cond: Expr::BinOp {
                lhs: Box::new(Expr::ThreadIdx {dim: Dim::X, ty: i64_ty(), i: i()}),
                op: BinOp::Eq,
                rhs: Box::new(int(0)),
                ty: Type::Boolean, i: i()
            },
            thn: vec![stmt],
            els: vec![],
            i: i()
        }
    }

    #[test]
    fn test_block_and_thread_loop() {
        let a_id = id("A");
        let b_id = id("B");
        let x_id = id("x");
        let y_id = id("y");
        let z_id = id("z");
        let i_id = id("i");
        let j_id = id("j");
        let a_idx1 = Expr::ArrayAccess {
            target: Box::new(var(a_id.clone())), idx: Box::new(var(i_id.clone())),
            ty: i64_ty(), i: i()
        };
        let a_idx2 = Expr::ArrayAccess {
            target: Box::new(var(a_id.clone())), idx: Box::new(var(j_id.clone())),
            ty: i64_ty(), i: i()
        };
        let b_idx = Expr::ArrayAccess {
            target: Box::new(var(b_id.clone())), idx: Box::new(var(i_id.clone())),
            ty: i64_ty(), i: i()
        };
        let stmts = vec![
            Stmt::Definition {ty: i64_ty(), id: x_id.clone(), expr: int(3), i: i()},
            block_dependent_loop(i_id.clone(), Dim::X, 0, 10, vec![
                Stmt::Assign {dst: var(y_id.clone()), expr: var(i_id.clone()), i: i()},
                thread_dependent_loop(j_id.clone(), Dim::X, 0, 10, vec![
                    Stmt::Assign {dst: var(z_id.clone()), expr: var(j_id.clone()), i: i()},
                    Stmt::Assign {dst: a_idx1.clone(), expr: int(1), i: i()},
                    Stmt::Assign {dst: a_idx2.clone(), expr: var(j_id.clone()), i: i()}
                ]),
                Stmt::Assign {dst: b_idx.clone(), expr: int(3), i: i()},
            ]),
        ];
        let vars = stmts.sfold(
            Ok(BTreeSet::new()),
            find_thread_index_dependent_variables_stmt
        ).unwrap();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&a_id));
        assert!(vars.contains(&j_id));
        assert!(vars.contains(&z_id));

        let stmts = transform_thread_independent_memory_writes_stmts(stmts, &vars);
        let expected = vec![
            Stmt::Definition {ty: i64_ty(), id: x_id.clone(), expr: int(3), i: i()},
            block_dependent_loop(i_id.clone(), Dim::X, 0, 10, vec![
                Stmt::Assign {dst: var(y_id.clone()), expr: var(i_id.clone()), i: i()},
                thread_dependent_loop(j_id.clone(), Dim::X, 0, 10, vec![
                    Stmt::Assign {dst: var(z_id), expr: var(j_id.clone()), i: i()},
                    tcheck(Stmt::Assign {dst: a_idx1, expr: int(1), i: i()}),
                    Stmt::SynchronizeBlock {i: i()},
                    Stmt::Assign {dst: a_idx2, expr: var(j_id.clone()), i: i()},
                ]),
                tcheck(Stmt::Assign {dst: b_idx, expr: int(3), i: i()}),
                Stmt::SynchronizeBlock {i: i()},
            ]),
        ];
        assert_eq!(stmts, expected);
    }
}
