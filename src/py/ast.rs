use crate::utils::info::*;
use crate::utils::name::Name;

use strum_macros::EnumIter;
use itertools::Itertools;

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, EnumIter)]
pub enum ElemSize {
    Bool, I8, I16, I32, I64, F16, F32, F64
}

impl ElemSize {
    pub fn is_signed_integer(&self) -> bool {
        match self {
            ElemSize::I8 | ElemSize::I16 | ElemSize::I32 | ElemSize::I64 => true,
            _ => false
        }
    }

    pub fn is_floating_point(&self) -> bool {
        match self {
            ElemSize::F16 | ElemSize::F32 | ElemSize::F64 => true,
            _ => false
        }
    }
}

impl fmt::Display for ElemSize {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ElemSize::Bool => write!(f, "bool"),
            ElemSize::I8 => write!(f, "int8"),
            ElemSize::I16 => write!(f, "int16"),
            ElemSize::I32 => write!(f, "int32"),
            ElemSize::I64 => write!(f, "int64"),
            ElemSize::F16 => write!(f, "float16"),
            ElemSize::F32 => write!(f, "float32"),
            ElemSize::F64 => write!(f, "float64"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Type {
    String,
    Tensor {sz: ElemSize, shape: Vec<i64>},
    Tuple {elems: Vec<Type>},
    Dict {fields: BTreeMap<String, Type>},
    Unknown
}

impl Type {
    pub fn get_scalar_elem_size<'a>(&'a self) -> Option<&'a ElemSize> {
        match self {
            Type::Tensor {sz, shape} if shape.len() == 0 => Some(sz),
            _ => None
        }
    }

    pub fn is_boolean(&self) -> bool {
        self.get_scalar_elem_size()
            .is_some_and(|sz| sz == &ElemSize::Bool)
    }

    pub fn is_signed_integer(&self) -> bool {
        self.get_scalar_elem_size()
            .is_some_and(|sz| sz.is_signed_integer())
    }

    pub fn is_floating_point(&self) -> bool {
        self.get_scalar_elem_size()
            .is_some_and(|sz| sz.is_floating_point())
    }

    pub fn get_dict_type_fields(&self) -> BTreeMap<String, Type> {
        if let Type::Dict {fields} = self {
            fields.clone()
        } else {
            panic!("Parir internal error: expected dictionary type, found {self}")
        }
    }
}

impl Ord for Type {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Type::String, Type::String) => Ordering::Equal,
            (Type::String, _) => Ordering::Less,
            (Type::Tensor {..}, Type::String) =>
                Ordering::Greater,
            (Type::Tensor {sz: lsz, shape: lsh}, Type::Tensor {sz: rsz, shape: rsh}) => {
                lsz.cmp(rsz).then(lsh.cmp(rsh))
            },
            (Type::Tensor {..}, _) => Ordering::Less,
            (Type::Tuple {..}, Type::Dict {..} | Type::Unknown) => Ordering::Less,
            (Type::Tuple {elems: lelems}, Type::Tuple {elems: relems}) =>
                lelems.cmp(relems),
            (Type::Tuple {..}, _) => Ordering::Greater,
            (Type::Dict {..}, Type::Unknown) => Ordering::Less,
            (Type::Dict {fields: lfields}, Type::Dict {fields: rfields}) =>
                lfields.iter()
                    .zip(rfields.iter())
                    .fold(Ordering::Equal, |acc, ((lk, lv), (rk, rv))| {
                        acc.then(lk.cmp(rk)).then(lv.cmp(rv))
                    }),
            (Type::Dict {..}, _) => Ordering::Greater,
            (Type::Unknown, Type::Unknown) => Ordering::Equal,
            (Type::Unknown, _) => Ordering::Greater,
        }
    }
}

impl PartialOrd for Type {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Type {}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Type::String => write!(f, "string"),
            Type::Tensor {sz, shape} if shape.is_empty() => write!(f, "{sz}"),
            Type::Tensor {sz, shape} => {
                let sh = shape.iter().map(|i| i.to_string()).join(",");
                write!(f, "tensor<{sz}>[{sh}]")
            },
            Type::Unknown => write!(f, "?"),
            Type::Tuple {elems} => {
                let elems = elems.iter()
                    .map(|e| format!("{e}"))
                    .join(",");
                write!(f, "({elems})")
            },
            Type::Dict {fields} => {
                let fields = fields.iter()
                    .map(|(k, v)| format!("{k} {v}"))
                    .join(",");
                write!(f, "dict {{{fields}}}")
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Builtin {
    Exp, Inf, Log, Max, Min, Abs, Cos, Sin, Sqrt, Tanh, Atan2,
    Convert {sz: ElemSize}, Label, GpuContext
}

impl fmt::Display for Builtin {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Builtin::Exp => write!(f, "exp"),
            Builtin::Inf => write!(f, "inf"),
            Builtin::Log => write!(f, "log"),
            Builtin::Max => write!(f, "max"),
            Builtin::Min => write!(f, "min"),
            Builtin::Abs => write!(f, "abs"),
            Builtin::Cos => write!(f, "cos"),
            Builtin::Sin => write!(f, "sin"),
            Builtin::Sqrt => write!(f, "sqrt"),
            Builtin::Tanh => write!(f, "tanh"),
            Builtin::Atan2 => write!(f, "atan2"),
            Builtin::Convert {sz} => write!(f, "convert({sz})"),
            Builtin::Label => write!(f, "<label>"),
            Builtin::GpuContext => write!(f, "gpu")
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum UnOp {
    Sub, Not, BitNeg
}

impl fmt::Display for UnOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            UnOp::Sub => write!(f, "-"),
            UnOp::Not => write!(f, "!"),
            UnOp::BitNeg => write!(f, "~"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BinOp {
    Add, Sub, Mul, FloorDiv, Div, Mod, Pow, And, Or,
    BitAnd, BitOr, BitXor, BitShl, BitShr, Eq, Neq, Leq, Geq, Lt, Gt
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::FloorDiv => write!(f, "//"),
            BinOp::Div => write!(f, "/"),
            BinOp::Mod => write!(f, "%"),
            BinOp::Pow => write!(f, "**"),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::BitAnd => write!(f, "&"),
            BinOp::BitOr => write!(f, "|"),
            BinOp::BitXor => write!(f, "^"),
            BinOp::BitShl => write!(f, "<<"),
            BinOp::BitShr => write!(f, ">>"),
            BinOp::Eq => write!(f, "=="),
            BinOp::Neq => write!(f, "!="),
            BinOp::Leq => write!(f, "<="),
            BinOp::Geq => write!(f, ">="),
            BinOp::Lt => write!(f, "<"),
            BinOp::Gt => write!(f, ">"),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Var {id: Name, ty: Type, i: Info},
    String {v: String, ty: Type, i: Info},
    Bool {v: bool, ty: Type, i: Info},
    Int {v: i64, ty: Type, i: Info},
    Float {v: f64, ty: Type, i: Info},
    UnOp {op: UnOp, arg: Box<Expr>, ty: Type, i: Info},
    BinOp {lhs: Box<Expr>, op: BinOp, rhs: Box<Expr>, ty: Type, i: Info},
    IfExpr {cond: Box<Expr>, thn: Box<Expr>, els: Box<Expr>, ty: Type, i: Info},
    Subscript {target: Box<Expr>, idx: Box<Expr>, ty: Type, i: Info},
    Tuple {elems: Vec<Expr>, ty: Type, i: Info},
    Dict {fields: BTreeMap<String, Expr>, ty: Type, i: Info},
    Builtin {func: Builtin, args: Vec<Expr>, ty: Type, i: Info},
    Convert {e: Box<Expr>, ty: Type},
}

impl Expr {
    pub fn get_type<'a>(&'a self) -> &'a Type {
        match self {
            Expr::Var {ty, ..} => ty,
            Expr::String {ty, ..} => ty,
            Expr::Bool {ty, ..} => ty,
            Expr::Int {ty, ..} => ty,
            Expr::Float {ty, ..} => ty,
            Expr::UnOp {ty, ..} => ty,
            Expr::BinOp {ty, ..} => ty,
            Expr::IfExpr {ty, ..} => ty,
            Expr::Subscript {ty, ..} => ty,
            Expr::Tuple {ty, ..} => ty,
            Expr::Dict {ty, ..} => ty,
            Expr::Builtin {ty, ..} => ty,
            Expr::Convert {ty, ..} => ty,
        }
    }

    pub fn discriminator(&self) -> u8 {
        match self {
            Expr::Var {..} => 0,
            Expr::String {..} => 1,
            Expr::Bool {..} => 2,
            Expr::Int {..} => 3,
            Expr::Float {..} => 4,
            Expr::UnOp {..} => 5,
            Expr::BinOp {..} => 6,
            Expr::IfExpr {..} => 7,
            Expr::Subscript {..} => 8,
            Expr::Tuple {..} => 9,
            Expr::Dict {..} => 10,
            Expr::Builtin {..} => 11,
            Expr::Convert {..} => 12
        }
    }

    pub fn with_info(self, i: Info) -> Self {
        match self {
            Expr::Var {id, ty, ..} => Expr::Var {id, ty, i},
            Expr::String {v, ty, ..} => Expr::String {v, ty, i},
            Expr::Bool {v, ty, ..} => Expr::Bool {v, ty, i},
            Expr::Int {v, ty, ..} => Expr::Int {v, ty, i},
            Expr::Float {v, ty, ..} => Expr::Float {v, ty, i},
            Expr::UnOp {op, arg, ty, ..} => Expr::UnOp {op, arg, ty, i},
            Expr::BinOp {lhs, op, rhs, ty, ..} => Expr::BinOp {lhs, op, rhs, ty, i},
            Expr::IfExpr {cond, thn, els, ty, ..} => Expr::IfExpr {cond, thn, els, ty, i},
            Expr::Subscript {target, idx, ty, ..} => Expr::Subscript {target, idx, ty, i},
            Expr::Tuple {elems, ty, ..} => Expr::Tuple {elems, ty, i},
            Expr::Dict {fields, ty, ..} => Expr::Dict {fields, ty, i},
            Expr::Builtin {func, args, ty, ..} => Expr::Builtin {func, args, ty, i},
            Expr::Convert {e, ty} => Expr::Convert {e: Box::new(e.with_info(i)), ty}
        }
    }
}

impl Ord for Expr {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Expr::Var {id: lid, ..}, Expr::Var {id: rid, ..}) => lid.cmp(rid),
            (Expr::String {v: lv, ..}, Expr::String {v: rv, ..}) => lv.cmp(rv),
            (Expr::Bool {v: lv, ..}, Expr::Bool {v: rv, ..}) => lv.cmp(rv),
            (Expr::Int {v: lv, ..}, Expr::Int {v: rv, ..}) => lv.cmp(rv),
            (Expr::Float {v: lv, ..}, Expr::Float {v: rv, ..}) => f64::total_cmp(lv, rv),
            (Expr::UnOp {op: lop, arg: larg, ..}, Expr::UnOp {op: rop, arg: rarg, ..}) =>
                lop.cmp(rop).then(larg.cmp(rarg)),
            ( Expr::BinOp {lhs: llhs, op: lop, rhs: lrhs, ..}
            , Expr::BinOp {lhs: rlhs, op: rop, rhs: rrhs, ..} ) =>
                llhs.cmp(rlhs).then(lop.cmp(rop)).then(lrhs.cmp(rrhs)),
            ( Expr::IfExpr {cond: lcond, thn: lthn, els: lels, ..}
            , Expr::IfExpr {cond: rcond, thn: rthn, els: rels, ..} ) =>
                lcond.cmp(rcond).then(lthn.cmp(rthn)).then(lels.cmp(rels)),
            ( Expr::Subscript {target: ltarget, idx: lidx, ..}
            , Expr::Subscript {target: rtarget, idx: ridx, ..} ) =>
                ltarget.cmp(rtarget).then(lidx.cmp(ridx)),
            (Expr::Tuple {elems: lelems, ..}, Expr::Tuple {elems: relems, ..}) =>
                lelems.cmp(relems),
            (Expr::Dict {fields: lfields, ..}, Expr::Dict {fields: rfields, ..}) =>
                lfields.cmp(rfields),
            ( Expr::Builtin {func: lfunc, args: largs, ..}
            , Expr::Builtin {func: rfunc, args: rargs, ..} ) =>
                lfunc.cmp(rfunc).then(largs.cmp(rargs)),
            (Expr::Convert {e: le, ty: lty}, Expr::Convert {e: re, ty: rty}) =>
                le.cmp(re).then(lty.cmp(rty)),
            (lhs, rhs) => lhs.discriminator().cmp(&rhs.discriminator())
        }
    }
}

impl PartialOrd for Expr {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Expr {}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Var {id, ..} => write!(f, "{0}", id.get_str()),
            Expr::String {v, ..} => write!(f, "\"{v}\""),
            Expr::Bool {v, ..} => write!(f, "{v}"),
            Expr::Int {v, ..} => write!(f, "{v}"),
            Expr::Float {v, ..} => write!(f, "{v}"),
            Expr::UnOp {op, arg, ..} => write!(f, "{op}{arg}"),
            Expr::BinOp {lhs, op, rhs, ..} => write!(f, "({lhs} {op} {rhs})"),
            Expr::IfExpr {cond, thn, els, ..} => write!(f, "({thn} if {cond} else {els})"),
            Expr::Subscript {target, idx, ..} => write!(f, "{target}[{idx}]"),
            Expr::Tuple {elems, ..} => {
                let elems = elems.iter()
                    .map(|e| format!("{e}"))
                    .join(",");
                write!(f, "({elems})")
            },
            Expr::Dict {fields, ..} => {
                let fields = fields.iter()
                    .map(|(k, v)| format!("{k}: {v}"))
                    .join(",");
                write!(f, "{{{fields}}}")
            },
            Expr::Builtin {func, args, ..} => {
                if args.is_empty() {
                    write!(f, "{func}")
                } else {
                    let args = args.iter()
                        .map(|a| format!("{a}"))
                        .join(",");
                    write!(f, "{func}({args})")
                }
            },
            Expr::Convert {e, ty} => {
                write!(f, "({ty}){e}")
            },
        }
    }
}

impl InfoNode for Expr {
    fn get_info(&self) -> Info {
        match self {
            Expr::Var {i, ..} => i.clone(),
            Expr::String {i, ..} => i.clone(),
            Expr::Bool {i, ..} => i.clone(),
            Expr::Int {i, ..} => i.clone(),
            Expr::Float {i, ..} => i.clone(),
            Expr::UnOp {i, ..} => i.clone(),
            Expr::BinOp {i, ..} => i.clone(),
            Expr::IfExpr {i, ..} => i.clone(),
            Expr::Subscript {i, ..} => i.clone(),
            Expr::Tuple {i, ..} => i.clone(),
            Expr::Dict {i, ..} => i.clone(),
            Expr::Builtin {i, ..} => i.clone(),
            Expr::Convert {e, ..} => e.get_info(),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Definition {ty: Type, id: Name, expr: Expr, i: Info},
    Assign {dst: Expr, expr: Expr, i: Info},
    For {var: Name, lo: Expr, hi: Expr, step: i64, body: Vec<Stmt>, i: Info},
    If {cond: Expr, thn: Vec<Stmt>, els: Vec<Stmt>, i: Info},
    While {cond: Expr, body: Vec<Stmt>, i: Info},
    WithGpuContext {body: Vec<Stmt>, i: Info},
    Label {label: String, assoc: Option<Box<Stmt>>, i: Info}
}

impl InfoNode for Stmt {
    fn get_info(&self) -> Info {
        match self {
            Stmt::Definition {i, ..} => i.clone(),
            Stmt::Assign {i, ..} => i.clone(),
            Stmt::For {i, ..} => i.clone(),
            Stmt::If {i, ..} => i.clone(),
            Stmt::While {i, ..} => i.clone(),
            Stmt::WithGpuContext {i, ..} => i.clone(),
            Stmt::Label {i, ..} => i.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub id: Name,
    pub ty: Type,
    pub i: Info
}

#[derive(Clone, Debug)]
pub struct FunDef {
    pub id: Name,
    pub params: Vec<Param>,
    pub body: Vec<Stmt>,
    pub i: Info
}
