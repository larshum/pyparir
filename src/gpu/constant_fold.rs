use super::ast::*;
use crate::utils::constant_fold::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::*;

impl CFExpr<Type> for Expr {
    fn mk_unop(op: UnOp, arg: Expr, ty: Type, i: Info) -> Expr {
        Expr::UnOp {op, arg: Box::new(arg), ty, i}
    }

    fn mk_binop(lhs: Expr, op: BinOp, rhs: Expr, ty: Type, i: Info) -> Expr {
        Expr::BinOp {lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i}
    }

    fn bool_expr(v: bool, ty: Type, i: Info) -> Expr {
        Expr::Bool {v, ty, i}
    }

    fn int_expr(v: i64, ty: Type, i: Info) -> Expr {
        Expr::Int {v, ty, i}
    }

    fn float_expr(v: f64, ty: Type, i: Info) -> Expr {
        Expr::Float {v, ty, i}
    }

    fn get_bool_value(&self) -> Option<bool> {
        match self {
            Expr::Bool {v, ..} => Some(*v),
            _ => None
        }
    }

    fn get_int_value(&self) -> Option<i64> {
        match self {
            Expr::Int {v, ..} => Some(*v),
            _ => None
        }
    }

    fn get_float_value(&self) -> Option<f64> {
        match self {
            Expr::Float {v, ..} => Some(*v),
            _ => None
        }
    }
}

impl CFType for Type {
    fn is_bool(&self) -> bool {
        *self == Type::Boolean
    }

    fn is_int(&self) -> bool {
        match self {
            Type::Scalar {sz} if sz.is_signed_integer() => true,
            _ => false
        }
    }

    fn is_float(&self) -> bool {
        match self {
            Type::Scalar {sz} if sz.is_floating_point() => true,
            _ => false
        }
    }
}

fn fold_expr(e: Expr) -> Expr {
    match e {
        Expr::UnOp {op, arg, ty, i} => {
            let arg = fold_expr(*arg);
            constant_fold_unop(op, arg, ty, i)
        },
        Expr::BinOp {lhs, op, rhs, ty, i} => {
            let lhs = fold_expr(*lhs);
            let rhs = fold_expr(*rhs);
            constant_fold_binop(lhs, op, rhs, ty, i)
        },
        Expr::Convert {e, ty} => {
            let e = fold_expr(*e);
            match e {
                Expr::Float {v, i, ..} if v.is_infinite() => {
                    Expr::Float {v, ty, i}
                },
                _ => Expr::Convert {e: Box::new(e), ty}
            }
        },
        Expr::Var {..} | Expr::Bool {..} | Expr::Int {..} | Expr::Float {..} |
        Expr::IfExpr {..} | Expr::StructFieldAccess {..} |
        Expr::ArrayAccess {..} | Expr::Struct {..} | Expr::WarpReduce {..} |
        Expr::ThreadIdx {..} | Expr::BlockIdx {..} => e.smap(fold_expr)
    }
}

fn replace_thread_block_indices_with_zero(e: Expr) -> Expr {
    match e {
        Expr::ThreadIdx {ty, i, ..} => Expr::Int {v: 0, ty, i},
        Expr::BlockIdx {ty, i, ..} => Expr::Int {v: 0, ty, i},
        _ => e.smap(replace_thread_block_indices_with_zero)
    }
}

fn is_zero_value(init: &Expr) -> bool {
    let init = fold_expr(replace_thread_block_indices_with_zero(init.clone()));
    match init {
        Expr::Int {v, ..} if v == 0 => true,
        _ => false
    }
}

fn cond_upper_bound(var: &Name, cond: &Expr) -> Option<i64> {
    match cond {
        Expr::BinOp {lhs, op: BinOp::Lt, rhs, ..} => {
            match (lhs.as_ref(), rhs.as_ref()) {
                (Expr::Var {id, ..}, Expr::Int {v, ..}) if id == var => Some(*v),
                _ => None
            }
        },
        _ => None
    }
}

fn incr_rhs(var: &Name, incr: &Expr) -> Option<i64> {
    match incr {
        Expr::BinOp {lhs, op: BinOp::Add, rhs, ..} => {
            match (lhs.as_ref(), rhs.as_ref()) {
                (Expr::Var {id, ..}, Expr::Int {v, ..}) if id == var => Some(*v),
                _ => None
            }
        },
        _ => None
    }
}

// If every iteration of a for-loop runs on a distinct thread (with a unique thread and block
// index) and each thread runs the for-loop exactly one iteration, we can eliminate the for-loop to
// improve readability of the generated code.
fn loop_runs_once(var: &Name, init: &Expr, cond: &Expr, incr: &Expr) -> bool {
    if is_zero_value(init) {
        match (cond_upper_bound(var, cond), incr_rhs(var, incr)) {
            (Some(l), Some(r)) if l == r => true,
            _ => false
        }
    } else {
        false
    }
}

enum LitBoolValue { True, False, Unknown }

fn literal_bool_value(cond: &Expr) -> LitBoolValue {
    match cond {
        Expr::Bool {v, ..} if *v => LitBoolValue::True,
        Expr::Bool {v, ..} if !*v => LitBoolValue::False,
        Expr::Int {v, ..} if *v != 0 => LitBoolValue::True,
        Expr::Int {v, ..} if *v == 0 => LitBoolValue::False,
        _ => LitBoolValue::Unknown
    }
}

fn fold_stmt_acc(mut acc: Vec<Stmt>, s: Stmt) -> Vec<Stmt> {
    match s {
        Stmt::For {var_ty, var, init, cond, incr, body, i} => {
            let init = fold_expr(init);
            let cond = fold_expr(cond);
            let incr = fold_expr(incr);
            if loop_runs_once(&var, &init, &cond, &incr) {
                acc.push(Stmt::Definition {ty: var_ty, id: var, expr: init, i});
                body.sfold_owned(acc, fold_stmt_acc)
            } else {
                let body = fold_stmts(body);
                acc.push(Stmt::For {var_ty, var, init, cond, incr, body, i});
                acc
            }
        },
        Stmt::If {cond, thn, els, i} => {
            let cond = fold_expr(cond);
            match literal_bool_value(&cond) {
                LitBoolValue::True => thn.sfold_owned(acc, fold_stmt_acc),
                LitBoolValue::False => els.sfold_owned(acc, fold_stmt_acc),
                LitBoolValue::Unknown => {
                    let thn = fold_stmts(thn);
                    let els = fold_stmts(els);
                    acc.push(Stmt::If {cond, thn, els, i});
                    acc
                }
            }
        },
        Stmt::While {cond, body, i} => {
            let cond = fold_expr(cond);
            let body = fold_stmts(body);
            acc.push(Stmt::While {cond, body, i});
            acc
        },
        Stmt::Scope {body, ..} => body.sfold_owned(acc, fold_stmt_acc),
        Stmt::Definition {..} | Stmt::Assign {..} |
        Stmt::SynchronizeBlock {..} | Stmt::KernelLaunch {..} |
        Stmt::Alloc {..} | Stmt::Dealloc {..} => {
            acc.push(s.smap(fold_expr));
            acc
        }
    }
}

fn fold_stmts(stmts: Vec<Stmt>) -> Vec<Stmt> {
    stmts.sfold_owned(vec![], fold_stmt_acc)
}

fn fold_top(top: Top) -> Top {
    match top {
        Top::DeviceFunDef {threads, id, params, body} => {
            let body = fold_stmts(body);
            Top::DeviceFunDef {threads, id, params, body}
        },
        Top::HostFunDef {ret_ty, id, params, body} => {
            let body = fold_stmts(body);
            Top::HostFunDef {ret_ty, id, params, body}
        },
        Top::StructDef {..} => top
    }
}

pub fn fold(ast: Ast) -> Ast {
    ast.into_iter().map(fold_top).collect::<Ast>()
}
