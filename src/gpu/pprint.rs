use super::ast::*;
use crate::utils::pprint::*;

use itertools::Itertools;

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::BTreeMap;

impl PrettyPrint for ElemSize {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            ElemSize::Bool => "bool",
            ElemSize::I8 => "int8_t",
            ElemSize::I16 => "int16_t",
            ElemSize::I32 => "int32_t",
            ElemSize::I64 => "int64_t",
            ElemSize::F16 => "half",
            ElemSize::F32 => "float",
            ElemSize::F64 => "double",
        };
        (env, s.to_string())
    }
}

impl<T: PrettyPrint> PrettyPrint for BTreeMap<String, T> {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let (env, s) = self.iter()
            .fold((env, vec![]), |(env, mut strs), (id, v)| {
                let (env, s) = v.pprint(env);
                let s = format!("{0}: {1}", id, s);
                strs.push(s);
                (env, strs)
            });
        (env, s.into_iter().join(", "))
    }
}

impl PrettyPrint for MemSpace {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            MemSpace::Device => "device",
            MemSpace::Shared => "shared",
        };
        (env, s.to_string())
    }
}

impl PrettyPrint for Dim {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let s = match self {
            Dim::X => "x",
            Dim::Y => "y",
            Dim::Z => "z",
        };
        (env, s.to_string())
    }
}

impl PrettyPrint for Type {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Type::Void => (env, "void".to_string()),
            Type::Boolean => (env, "bool".to_string()),
            Type::Scalar {sz} => sz.pprint(env),
            Type::Pointer {ty, mem} => {
                let (env, ty) = ty.pprint(env);
                let (env, mem) = mem.pprint(env);
                (env, format!("{mem} {ty}*"))
            },
            Type::Struct {id, ..} => {
                let (env, id) = id.pprint(env);
                (env, format!("{id}"))
            },
        }
    }
}

pub fn print_unop(op: &UnOp, ty: &Type) -> String {
    let s = match op {
        UnOp::Sub => "-",
        UnOp::Not => "!",
        UnOp::BitNeg => "~",
        UnOp::Exp => match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "hexp",
            Some(ElemSize::F32) => "__expf",
            Some(ElemSize::F64) => "exp",
            _ => panic!("Invalid type of exp")
        },
        UnOp::Log => match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "hlog",
            Some(ElemSize::F32) => "__logf",
            Some(ElemSize::F64) => "log",
            _ => panic!("Invalid type of log")
        },
        UnOp::Cos => match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "hcos",
            Some(ElemSize::F32) => "__cosf",
            Some(ElemSize::F64) => "cos",
            _ => panic!("Invalid type of cos")
        },
        UnOp::Sin => match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "hsin",
            Some(ElemSize::F32) => "__sinf",
            Some(ElemSize::F64) => "sin",
            _ => panic!("Invalid type of sin")
        },
        UnOp::Sqrt => match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "hsqrt",
            Some(ElemSize::F32) => "sqrtf",
            Some(ElemSize::F64) => "sqrt",
            _ => panic!("Invalid type of sqrt")
        },
        UnOp::Tanh => match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "htanh",
            Some(ElemSize::F32) => "tanhf",
            Some(ElemSize::F64) => "tanh",
            _ => panic!("Invalid type of tanh")
        },
        UnOp::Abs => match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "__habs",
            Some(ElemSize::F32) => "fabsf",
            Some(ElemSize::F64) => "fabs",
            Some(_) => "abs",
            None => panic!("Invalid type of abs")
        },
    };
    s.to_string()
}

pub fn print_binop(op: &BinOp, argty: &Type, ty: &Type) -> String {
    let s = match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::FloorDiv | BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Pow => match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "hpow",
            Some(ElemSize::F32) => "__powf",
            Some(ElemSize::F64) => "pow",
            _ => panic!("Invalid type of **")
        },
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::BitAnd => "&",
        BinOp::BitOr => "|",
        BinOp::BitXor => "^",
        BinOp::BitShl => "<<",
        BinOp::BitShr => ">>",
        BinOp::Eq => match argty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "__heq",
            _ => "=="
        },
        BinOp::Neq => match argty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "__hne",
            _ => "!="
        },
        BinOp::Leq => match argty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "__hle",
            _ => "<="
        }
        BinOp::Geq => match argty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "__hge",
            _ => ">="
        },
        BinOp::Lt => match argty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "__hlt",
            _ => "<"
        },
        BinOp::Gt => match argty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "__hgt",
            _ => ">"
        },
        BinOp::Max => match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "__hmax",
            Some(ElemSize::F32) => "fmaxf",
            Some(ElemSize::F64) => "fmax",
            Some(_) => "max",
            None => panic!("Invalid type of max")
        },
        BinOp::Min => match ty.get_scalar_elem_size() {
            Some(ElemSize::F16) => "__hmin",
            Some(ElemSize::F32) => "fminf",
            Some(ElemSize::F64) => "fmin",
            Some(_) => "min",
            None => panic!("Invalid type of min")
        },
        BinOp::Atan2 => "atan2",
    };
    s.to_string()
}

fn try_get_binop(e: &Box<Expr>) -> Option<BinOp> {
    match e.borrow() {
        Expr::BinOp {op, ..} => Some(op.clone()),
        _ => None
    }
}

fn parenthesize_if_pred(
    inner_op_opt: Option<BinOp>,
    outer_op: &BinOp,
    s: String,
    pred: impl Fn(Ordering) -> bool
) -> String {
    match inner_op_opt {
        Some(inner_op) => {
            if pred(inner_op.precedence().cmp(&outer_op.precedence())) {
                format!("({s})")
            } else {
                s
            }
        },
        None => s
    }
}

fn parenthesize_if_lower_precedence(
    inner_op_opt: Option<BinOp>,
    outer_op: &BinOp,
    s: String
) -> String {
    parenthesize_if_pred(inner_op_opt, outer_op, s, |p| p == Ordering::Less)
}

fn parenthesize_if_lower_or_same_precedence(
    inner_op_opt: Option<BinOp>,
    outer_op: &BinOp,
    s: String
) -> String {
    parenthesize_if_pred(inner_op_opt, outer_op, s, |p| p != Ordering::Greater)
}

fn is_infix(op: &BinOp, ty: &Type) -> bool {
    let is_f16 = match ty.get_scalar_elem_size() {
        Some(ElemSize::F16) => true,
        _ => false
    };
    match op {
        BinOp::Pow | BinOp::Max | BinOp::Min | BinOp::Atan2 => false,
        BinOp::Eq | BinOp::Neq | BinOp::Leq | BinOp::Geq | BinOp::Lt |
        BinOp::Gt if is_f16 => false,
        _ => true
    }
}

impl PrettyPrint for Expr {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Expr::Var {id, ..} => id.pprint(env),
            Expr::Bool {v, ..} => (env, v.to_string()),
            Expr::Int {v, ..} => (env, v.to_string()),
            Expr::Float {v, ty, ..} => {
                if v.is_infinite() {
                    let s = match ty.get_scalar_elem_size() {
                        Some(ElemSize::F16) => "CUDART_INF_FP16",
                        Some(ElemSize::F32) => "HUGE_VALF",
                        Some(ElemSize::F64) => "HUGE_VAL",
                        _ => panic!("Invalid type of floating-point literal")
                    };
                    if v.is_sign_negative() {
                        (env, format!("-{s}"))
                    } else {
                        (env, s.to_string())
                    }
                } else {
                    // Debug printing adds a trailing '.0' for floats with no decimal component,
                    // which is what we want to have to distinguish them.
                    (env, format!("{v:?}"))
                }
            },
            Expr::UnOp {op, arg, ty, ..} => {
                let op_str = print_unop(&op, &ty);
                let (env, arg_str) = arg.pprint(env);
                (env, format!("{op_str}({arg_str})"))
            },
            Expr::BinOp {lhs, op, rhs, ty, ..} => {
                let (env, lhs_str) = lhs.pprint(env);
                let op_str = print_binop(&op, lhs.get_type(), &ty);
                let (env, rhs_str) = rhs.pprint(env);

                // We consider the precedence of the left- and right-hand side operands and use
                // this to omit parentheses when they are unnecessary.
                if is_infix(&op, lhs.get_type()) {
                    let lhs_op = try_get_binop(lhs);
                    let rhs_op = try_get_binop(rhs);
                    let lhs_str = parenthesize_if_lower_precedence(lhs_op, &op, lhs_str);
                    let rhs_str = parenthesize_if_lower_or_same_precedence(rhs_op, &op, rhs_str);
                    (env, format!("{lhs_str} {op_str} {rhs_str}"))
                } else {
                    (env, format!("{op_str}({lhs_str}, {rhs_str})"))
                }
            },
            Expr::IfExpr {cond, thn, els, ..} => {
                let (env, cond) = cond.pprint(env);
                let (env, thn) = thn.pprint(env);
                let (env, els) = els.pprint(env);
                (env, format!("({cond} ? {thn} : {els})"))
            },
            Expr::StructFieldAccess {target, label, ..} => {
                let (env, target) = target.pprint(env);
                (env, format!("{target}.{label}"))
            },
            Expr::ArrayAccess {target, idx, ..} => {
                let (env, target) = target.pprint(env);
                let (env, idx) = idx.pprint(env);
                (env, format!("{target}[{idx}]"))
            },
            Expr::Convert {e, ty} => {
                let (env, e_str) = e.pprint(env);
                let (env, ty) = ty.pprint(env);
                let s = if e.is_leaf_node() {
                    format!("({ty}){e_str}")
                } else {
                    format!("({ty})({e_str})")
                };
                (env, s)
            },
            Expr::Struct {id, fields, ..} => {
                let (env, id) = id.pprint(env);
                let (env, fields) = fields.iter()
                    .fold((env, vec![]), |(env, mut strs), (id, e)| {
                        let (env, e) = e.pprint(env);
                        strs.push(format!(".{id}: {e}"));
                        (env, strs)
                    });
                let outer_indent = env.print_indent();
                let env = env.incr_indent();
                let indent = env.print_indent();
                let fields = fields.into_iter().join(&format!(",\n{indent}"));
                let env = env.decr_indent();
                (env, format!("{id} {{\n{indent}{fields}\n{outer_indent}}}"))
            },
            Expr::WarpReduce {value, op, ..} => {
                let (env, value) = value.pprint(env);
                let (env, op) = op.pprint(env);
                (env, format!("warp_reduce({value}, {op})"))
            },
            Expr::ThreadIdx {dim, ..} => {
                let (env, dim) = dim.pprint(env);
                (env, format!("threadIdx.{dim}"))
            },
            Expr::BlockIdx {dim, ..} => {
                let (env, dim) = dim.pprint(env);
                (env, format!("blockIdx.{dim}"))
            },
        }
    }
}

impl PrettyPrint for Dim3 {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Dim3 {x, y, z} = self;
        (env, format!("{x}, {y}, {z}"))
    }
}

impl PrettyPrint for LaunchArgs {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let LaunchArgs {blocks, threads} = self;
        let (env, blocks) = blocks.pprint(env);
        let (env, threads) = threads.pprint(env);
        let indent = env.print_indent();
        (env, format!("{indent}{{blocks: ({blocks}), threads: ({threads})}};"))
    }
}

impl PrettyPrint for Stmt {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let indent = env.print_indent();
        match self {
            Stmt::Definition {ty, id, expr, ..} => {
                let (env, ty) = ty.pprint(env);
                let (env, id) = id.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{ty} {id} = {expr};"))
            },
            Stmt::Assign {dst, expr, ..} => {
                let (env, dst) = dst.pprint(env);
                let (env, expr) = expr.pprint(env);
                (env, format!("{indent}{dst} = {expr};"))
            },
            Stmt::For {var_ty, var, init, cond, incr, body, ..} => {
                let (env, var_ty) = var_ty.pprint(env);
                let (env, var) = var.pprint(env);
                let (env, init) = init.pprint(env);
                let (env, cond) = cond.pprint(env);
                let (env, incr) = incr.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let s = format!(
                    "{0}for ({1} {2} = {3}; {4}; {2} = {5}) {{\n{6}\n{0}}}",
                    indent, var_ty, var, init, cond, incr, body
                );
                (env, s)
            },
            Stmt::If {cond, thn, els, ..} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, thn_str) = pprint_iter(thn.iter(), env, "\n");
                let (env, thn, els) = match &els[..] {
                    [Stmt::If {
                        cond: elif_cond, thn: elif_thn, els: elif_els, ..
                    }] if !elif_thn.is_empty() => {
                        let (env, elif_cond) = elif_cond.pprint(env);
                        let (env, elif_thn) = pprint_iter(elif_thn.iter(), env, "\n");
                        let s = format!(
                            "{0}\n{1}}} else if ({2}) {{\n{3}",
                            thn_str, indent, elif_cond, elif_thn
                        );
                        (env, s, elif_els)
                    },
                    _ => (env, thn_str, els)
                };
                let (env, s) = if els.is_empty() {
                    let s = format!(
                        "{0}if ({1}) {{\n{2}\n{0}}}",
                        indent, cond, thn
                    );
                    (env, s)
                } else {
                    let (env, els) = pprint_iter(els.iter(), env, "\n");
                    let s = format!(
                        "{0}if ({1}) {{\n{2}\n{0}}} else {{\n{3}\n{0}}}",
                        indent, cond, thn, els
                    );
                    (env, s)
                };
                let env = env.decr_indent();
                (env, s)
            },
            Stmt::While {cond, body, ..} => {
                let (env, cond) = cond.pprint(env);
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                let s = format!("{0}while ({1}) {{\n{2}\n{0}}}", indent, cond, body);
                (env, s)
            },
            Stmt::Scope {body, ..} => {
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{0}{{\n{1}\n{0}}}", indent, body))
            },
            Stmt::SynchronizeBlock {..} => {
                (env, format!("{indent}sync_block();"))
            },
            Stmt::KernelLaunch {id, args, grid, ..} => {
                let (env, id) = id.pprint(env);
                let (env, args) = pprint_iter(args.iter(), env, ", ");
                let (env, grid) = grid.pprint(env);
                (env, format!("{indent}launch({id}, [{args}], {grid})"))
            },
            Stmt::Alloc {id, elem_ty, sz, mem, ..} => {
                let (env, id) = id.pprint(env);
                let (env, elem_ty) = elem_ty.pprint(env);
                let (env, mem) = mem.pprint(env);
                (env, format!("{indent}id = alloc({sz}, {elem_ty}, {mem});"))
            },
            Stmt::Dealloc {id, mem, ..} => {
                let (env, id) = id.pprint(env);
                let (env, mem) = mem.pprint(env);
                (env, format!("{indent}dealloc({id}, {mem});"))
            },
        }
    }
}

impl PrettyPrint for Param {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Param {id, ty, ..} = self;
        let (env, id) = id.pprint(env);
        let restrict_str = if let Type::Pointer {..} = &ty {
            " __restrict__"
        } else {
            ""
        };
        let (env, ty) = ty.pprint(env);
        (env, format!("{ty}{restrict_str} {id}"))
    }
}

impl PrettyPrint for Field {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        let Field {id, ty, ..} = self;
        let (env, ty) = ty.pprint(env);
        let indent = env.print_indent();
        (env, format!("{indent}{ty} {id};"))
    }
}

impl PrettyPrint for Top {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        match self {
            Top::DeviceFunDef {threads, id, params, body} => {
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("threads({threads})\nvoid {id}({params}) {{\n{body}\n}}"))
            },
            Top::HostFunDef {ret_ty, id, params, body} => {
                let (env, ret_ty) = ret_ty.pprint(env);
                let (env, id) = id.pprint(env);
                let (env, params) = pprint_iter(params.iter(), env, ", ");
                let env = env.incr_indent();
                let (env, body) = pprint_iter(body.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("{ret_ty} {id}({params}) {{\n{body}\n}}"))
            },
            Top::StructDef {id, fields} => {
                let (env, id) = id.pprint(env);
                let env = env.incr_indent();
                let (env, fields) = pprint_iter(fields.iter(), env, "\n");
                let env = env.decr_indent();
                (env, format!("struct {id} {{\n{fields}\n}};"))
            },
        }
    }
}

impl PrettyPrint for Ast {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        pprint_iter(self.iter(), env, "\n")
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::utils::info::Info;
    use crate::utils::name::Name;
    use crate::utils::pprint;

    fn var(s: &str) -> Expr {
        Expr::Var {id: Name::new(s.to_string()), ty: Type::Boolean, i: Info::default()}
    }

    fn int(v: i64) -> Expr {
        Expr::Int {v, ty: Type::Scalar {sz: ElemSize::I64}, i: Info::default()}
    }

    fn bop(lhs: Expr, op: BinOp, rhs: Expr, ty: Option<Type>) -> Expr {
        let ty = ty.unwrap_or(Type::Boolean);
        Expr::BinOp {
            lhs: Box::new(lhs), op, rhs: Box::new(rhs), ty, i: Info::default()
        }
    }

    fn unop(op: UnOp, arg: Expr, ty: Type) -> Expr {
        Expr::UnOp {op, arg: Box::new(arg), ty, i: Info::default()}
    }

    fn add(lhs: Expr, rhs: Expr) -> Expr {
        bop(lhs, BinOp::Add, rhs, None)
    }

    fn mul(lhs: Expr, rhs: Expr) -> Expr {
        bop(lhs, BinOp::Mul, rhs, None)
    }

    fn rem(lhs: Expr, rhs: Expr) -> Expr {
        bop(lhs, BinOp::Rem, rhs, None)
    }

    fn scalar_ty(sz: ElemSize) -> Type {
        Type::Scalar {sz}
    }

    fn int64_ty() -> Type {
        scalar_ty(ElemSize::I64)
    }

    fn i() -> Info {
        Info::default()
    }

    #[test]
    fn pprint_precedence_same_level_with_paren() {
        let s = add(var("x"), add(var("y"), var("z"))).pprint_default();
        assert_eq!(&s, "x + (y + z)");
    }

    #[test]
    fn pprint_precedence_same_level_omit_paren() {
        let s = add(add(var("x"), var("y")), var("z")).pprint_default();
        assert_eq!(&s, "x + y + z");
    }

    #[test]
    fn pprint_precedence_print_paren() {
        let s = mul(add(var("x"), var("y")), add(var("y"), var("z"))).pprint_default();
        assert_eq!(&s, "(x + y) * (y + z)");
    }

    #[test]
    fn pprint_precedence_rhs_paren() {
        let s = add(var("x"), add(mul(var("y"), var("y")), var("z"))).pprint_default();
        assert_eq!(&s, "x + (y * y + z)");
    }

    #[test]
    fn pprint_precedence_same_level_paren() {
        let s = mul(var("x"), rem(var("y"), var("z"))).pprint_default();
        assert_eq!(&s, "x * (y % z)");
    }

    #[test]
    fn pprint_dims() {
        assert_eq!(Dim::X.pprint_default(), "x");
        assert_eq!(Dim::Y.pprint_default(), "y");
        assert_eq!(Dim::Z.pprint_default(), "z");
    }

    #[test]
    fn pprint_thread_idx_x() {
        let s = Expr::ThreadIdx {dim: Dim::X, ty: int64_ty(), i: Info::default()}.pprint_default();
        assert_eq!(&s, "threadIdx.x");
    }

    #[test]
    fn pprint_block_idx_y() {
        let s = Expr::BlockIdx {dim: Dim::Y, ty: int64_ty(), i: Info::default()}.pprint_default();
        assert_eq!(&s, "blockIdx.y");
    }

    fn exp(arg: Expr, ty: Type) -> Expr {
        unop(UnOp::Exp, arg, ty)
    }

    fn log(arg: Expr, ty: Type) -> Expr {
        unop(UnOp::Log, arg, ty)
    }

    fn max(lhs: Expr, rhs: Expr, ty: Type) -> Expr {
        bop(lhs, BinOp::Max, rhs, Some(ty))
    }

    #[test]
    fn pprint_exp_f32() {
        let s = exp(var("x"), scalar_ty(ElemSize::F32)).pprint_default();
        assert_eq!(&s, "__expf(x)");
    }

    #[test]
    #[should_panic]
    fn pprint_exp_int_type_fails() {
        exp(var("x"), scalar_ty(ElemSize::I32)).pprint_default();
    }

    #[test]
    #[should_panic]
    fn pprint_exp_invalid_type_fails() {
        exp(var("x"), Type::Boolean).pprint_default();
    }

    #[test]
    fn pprint_log_f64() {
        let s = log(var("x"), scalar_ty(ElemSize::F64)).pprint_default();
        assert_eq!(&s, "log(x)");
    }

    #[test]
    fn pprint_max_f32() {
        let s = max(var("x"), var("y"), scalar_ty(ElemSize::F32)).pprint_default();
        assert_eq!(&s, "fmaxf(x, y)");
    }

    #[test]
    fn pprint_max_i64() {
        let s = max(var("x"), var("y"), scalar_ty(ElemSize::I64)).pprint_default();
        assert_eq!(&s, "max(x, y)");
    }

    fn convert(e: Expr, ty: Type) -> Expr {
        Expr::Convert {e: Box::new(e), ty}
    }

    #[test]
    fn pprint_var_conversion() {
        let s = convert(var("x"), scalar_ty(ElemSize::F32));
        assert_eq!(&s.pprint_default(), "(float)x");
    }

    #[test]
    fn pprint_literal_conversion() {
        let s = convert(int(5), scalar_ty(ElemSize::I16));
        assert_eq!(&s.pprint_default(), "(int16_t)5");
    }

    #[test]
    fn pprint_add_conversion() {
        let s = convert(add(var("x"), var("y")), scalar_ty(ElemSize::I16));
        assert_eq!(&s.pprint_default(), "(int16_t)(x + y)");
    }

    #[test]
    fn pprint_launch_args() {
        let args = LaunchArgs::default()
            .with_blocks_dim(&Dim::Y, 2)
            .with_blocks_dim(&Dim::Z, 3)
            .with_threads_dim(&Dim::X, 4)
            .with_threads_dim(&Dim::Y, 5)
            .with_threads_dim(&Dim::Z, 6);
        let s = args.pprint_default();
        assert_eq!(&s, "{blocks: (1, 2, 3), threads: (4, 5, 6)};");
    }

    #[test]
    fn pprint_syncblock() {
        let s = Stmt::SynchronizeBlock {i: i()}.pprint_default();
        assert_eq!(&s, "sync_block();");
    }

    #[test]
    fn pprint_for_loop() {
        let id = Name::new("i".to_string());
        let i_var = Expr::Var {id: id.clone(), ty: scalar_ty(ElemSize::I64), i: i()};
        let for_loop = Stmt::For {
            var_ty: scalar_ty(ElemSize::I64),
            var: id,
            init: int(0),
            cond: bop(i_var.clone(), BinOp::Lt, int(10), None),
            incr: bop(i_var.clone(), BinOp::Add, int(1), None),
            body: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}],
            i: i()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!(
            "for (int64_t i = 0; i < 10; i = i + 1) {{\n{indent}x = y;\n}}"
        );
        assert_eq!(for_loop.pprint_default(), expected);
    }

    #[test]
    fn pprint_if_cond() {
        let cond = Stmt::If {
            cond: Expr::BinOp {
                lhs: Box::new(var("x")),
                op: BinOp::Eq,
                rhs: Box::new(var("y")),
                ty: Type::Boolean,
                i: i()
            },
            thn: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}],
            els: vec![Stmt::Assign {dst: var("y"), expr: var("x"), i: i()}],
            i: i()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!(
            "if (x == y) {{\n{0}x = y;\n}} else {{\n{0}y = x;\n}}", indent
        );
        assert_eq!(cond.pprint_default(), expected);
    }

    #[test]
    fn pprint_if_cond_empty_else() {
        let cond = Stmt::If {
            cond: Expr::BinOp {
                lhs: Box::new(var("x")),
                op: BinOp::Eq,
                rhs: Box::new(var("y")),
                ty: Type::Boolean,
                i: i()
            },
            thn: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}],
            els: vec![],
            i: i()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("if (x == y) {{\n{indent}x = y;\n}}");
        assert_eq!(cond.pprint_default(), expected);
    }

    #[test]
    fn pprint_if_cond_elseif() {
        let cond = Stmt::If {
            cond: var("x"),
            thn: vec![Stmt::Assign {dst: var("y"), expr: var("z"), i: i()}],
            els: vec![Stmt::If {
                    cond: var("y"),
                    thn: vec![Stmt::Assign {dst: var("x"), expr: var("z"), i: i()}],
                    els: vec![Stmt::Assign {dst: var("z"), expr: var("x"), i: i()}],
                    i: i()
            }],
            i: i()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!(
            "if (x) {{\n{0}y = z;\n}} else if (y) {{\n{0}x = z;\n}} else {{\n{0}z = x;\n}}",
            indent
        );
        assert_eq!(cond.pprint_default(), expected);
    }

    #[test]
    fn pprint_while() {
        let wh = Stmt::While {
            cond: var("x"),
            body: vec![Stmt::Assign {dst: var("y"), expr: var("z"), i: i()}],
            i: i()
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("while (x) {{\n{indent}y = z;\n}}");
        assert_eq!(wh.pprint_default(), expected);
    }

    #[test]
    fn pprint_kernel_launch() {
        let id = "kernel";
        let kernel = Stmt::KernelLaunch {
            id: Name::new(id.to_string()),
            args: vec![var("x"), var("y")],
            grid: LaunchArgs::default(),
            i: i()
        };
        let grid_str = LaunchArgs::default().pprint_default();
        let expected = format!("launch({id}, [x, y], {grid_str})");
        assert_eq!(kernel.pprint_default(), expected);
    }

    #[test]
    fn pprint_device_fun_def() {
        let def = Top::DeviceFunDef {
            threads: 1024,
            id: Name::new("f".to_string()),
            params: vec![],
            body: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}]
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("threads(1024)\nvoid f() {{\n{0}x = y;\n}}", indent);
    }

    #[test]
    fn pprint_host_fun_def() {
        let def = Top::HostFunDef {
            ret_ty: Type::Void,
            id: Name::new("f".to_string()),
            params: vec![],
            body: vec![Stmt::Assign {dst: var("x"), expr: var("y"), i: i()}]
        };
        let indent = " ".repeat(pprint::DEFAULT_INDENT);
        let expected = format!("void f() {{\n{0}x = y;\n}}", indent);
    }
}

