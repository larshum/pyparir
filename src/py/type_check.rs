use crate::py_type_error;
use crate::utils::err::*;
use crate::utils::info::*;
use crate::utils::name::Name;
use super::ast::*;

use pyo3::PyTypeInfo;
use pyo3::prelude::*;
use pyo3::types::*;
use pyo3::types::IntoPyDict;

use std::collections::BTreeMap;

use itertools::Itertools;

fn compile_elem_size<'py>(dtype: Bound<'py, PyAny>) -> PyResult<ElemSize> {
    let torch = dtype.py().import("torch")?;
    if dtype.eq(torch.getattr("bool")?)? {
        Ok(ElemSize::Bool)
    } else if dtype.eq(torch.getattr("int8")?)? {
        Ok(ElemSize::I8)
    } else if dtype.eq(torch.getattr("int16")?)? {
        Ok(ElemSize::I16)
    } else if dtype.eq(torch.getattr("int32")?)? {
        Ok(ElemSize::I32)
    } else if dtype.eq(torch.getattr("int64")?)? {
        Ok(ElemSize::I64)
    } else if dtype.eq(torch.getattr("float16")?)? {
        Ok(ElemSize::F16)
    } else if dtype.eq(torch.getattr("float32")?)? {
        Ok(ElemSize::F32)
    } else if dtype.eq(torch.getattr("float64")?)? {
        Ok(ElemSize::F64)
    } else {
        py_type_error!(Info::default(), "Unsupported element type: {dtype}")
    }
}

fn get_tensor_shape<'py>(
    t: &Bound<'py, PyAny>
) -> PyResult<Vec<i64>> {
    let py = t.py();
    let ndims = t.getattr("ndim")?.extract::<i64>()?;
    (0..ndims).into_iter()
        .map(|i| {
            let kwargs = [("dim", i)].into_py_dict(py)?;
            t.call_method("size", (), Some(&kwargs))?.extract::<i64>()
        })
        .collect::<PyResult<Vec<i64>>>()
}

fn convert_type<'py>(arg: &Bound<'py, PyAny>) -> PyResult<Type> {
    let py = arg.py();
    let torch = py.import("torch")?;
    let ty = arg.get_type();
    if ty.eq(torch.getattr("Tensor")?)? {
        let dtype = arg.getattr("dtype")?;
        let sz = compile_elem_size(dtype)?;
        let shape = get_tensor_shape(&arg)?;
        Ok(Type::Tensor {sz, shape})
    } else if arg.is_instance(&PyInt::type_object(arg.py()))? {
        Ok(Type::Tensor {sz: ElemSize::I64, shape: vec![]})
    } else if arg.is_instance(&PyFloat::type_object(arg.py()))? {
        Ok(Type::Tensor {sz: ElemSize::F64, shape: vec![]})
    } else if arg.is_instance(&PyDict::type_object(arg.py()))? {
        let fields = arg.call_method0("items")?
            .try_iter()?
            .map(|f| {
                let f = f?;
                let id = f.get_item(0)?.extract::<String>()?;
                let ty = f.get_item(1)?;
                Ok((id, convert_type(&ty)?))
            })
            .collect::<PyResult<BTreeMap<String, Type>>>()?;
        Ok(Type::Dict {fields})
    } else {
        py_type_error!(Info::default(), "Argument {0:?} has unsupported type {1:?}", arg, ty)
    }
}

fn add_param_types<'py>(
    id: &Name,
    params: Vec<Param>,
    args: &Vec<Bound<'py, PyAny>>
) -> PyResult<Vec<Param>> {
    if args.len() == params.len() {
        args.iter()
            .zip(params.into_iter())
            .map(|(arg, Param {id, i, ..})| Ok(Param {id, ty: convert_type(&arg)?, i}))
            .collect::<PyResult<Vec<Param>>>()
    } else {
        py_type_error!(Info::default(), "Function {id} expected {0} arguments but received {1}", params.len(), args.len())
    }
}

fn lub_elem_size(
    lhs: &ElemSize,
    rhs: &ElemSize,
    i: &Info
) -> PyResult<ElemSize> {
    match (lhs, rhs) {
        (ElemSize::Bool, ElemSize::Bool) => Ok(rhs.clone()),
        (ElemSize::I8, _) if rhs.is_signed_integer() => Ok(rhs.clone()),
        (ElemSize::I16, ElemSize::I8) => Ok(ElemSize::I16),
        (ElemSize::I16, _) if rhs.is_signed_integer() => Ok(rhs.clone()),
        (ElemSize::I32, ElemSize::I8 | ElemSize::I16) => Ok(ElemSize::I32),
        (ElemSize::I32, _) if rhs.is_signed_integer() => Ok(rhs.clone()),
        (ElemSize::I64, _) if rhs.is_signed_integer() => Ok(lhs.clone()),
        (ElemSize::F16, _) if rhs.is_floating_point() => Ok(rhs.clone()),
        (ElemSize::F32, ElemSize::F16) => Ok(ElemSize::F32),
        (ElemSize::F32, _) if rhs.is_floating_point() => Ok(rhs.clone()),
        (ElemSize::F64, _) if rhs.is_floating_point() => Ok(lhs.clone()),
        _ => py_type_error!(i, "Incompatible element types")
    }
}

fn compatible_elem_types(lhs: &ElemSize, rhs: &ElemSize) -> bool {
    lub_elem_size(lhs, rhs, &Info::default()).is_ok()
}

fn ensure_scalar_type(e: Expr, expected: ElemSize) -> PyResult<Expr> {
    let i = e.get_info();
    let ty = e.get_type();
    if let Some(actual) = ty.get_scalar_elem_size() {
        // We allow it if the two element size types are compatible. If the types are not
        // equivalent, we insert a conversion to the expected type.
        if compatible_elem_types(&actual, &expected) {
            if actual.eq(&expected) {
                Ok(e)
            } else {
                Ok(Expr::Convert {e: Box::new(e), ty: Type::Tensor {sz: expected, shape: vec![]}})
            }
        } else {
            py_type_error!(i, "Expected element of type {expected}, found incompatible element type {actual}")
        }
    } else {
        py_type_error!(i, "Expected element of type {expected}, found type {ty}")
    }
}

fn assert_type(e: &Expr, expected: &Type) -> PyResult<()> {
    let i = e.get_info();
    let actual = e.get_type();
    if actual.eq(expected) {
        Ok(())
    } else {
        py_type_error!(i, "Expected type {expected}, found type {actual}")
    }
}

fn coerce_type(e: Expr, expected: &Type) -> PyResult<Expr> {
    if let Ok(()) = assert_type(&e, expected) {
        Ok(e)
    } else {
        let i = e.get_info();
        let actual = e.get_type();
        match (actual, expected) {
            (Type::Tensor {sz: lsz, shape: lsh}, Type::Tensor {sz: rsz, shape: rsh}) => {
                if lsh.len() == 0 && rsh.len() == 0 {
                    ensure_scalar_type(e, rsz.clone())
                } else if lsz == rsz && lsh == rsh {
                    Ok(e)
                } else {
                    py_type_error!(i, "Cannot coerce incompatible tensor types ({actual} != {expected})")
                }
            }
            (Type::Tuple {..}, Type::Tuple {elems: r}) => {
                if let Expr::Tuple {elems, i, ..} = e {
                    let elems = elems.into_iter()
                        .zip(r.iter())
                        .map(|(e, ty)| coerce_type(e, ty))
                        .collect::<PyResult<Vec<Expr>>>()?;
                    let elem_tys = elems.iter()
                        .map(|e| e.get_type().clone())
                        .collect::<Vec<Type>>();
                    let ty = Type::Tuple {elems: elem_tys};
                    Ok(Expr::Tuple {elems, ty, i})
                } else {
                    py_type_error!(i, "Cannot coerce non-literal tuple value {e}")
                }
            },
            _ => py_type_error!(i, "Cannot coerce expression {e} of type {actual} to type {expected}")
        }
    }
}

/// Finds the least upper bound of two types, with respect to sizes of types. The least upper bound
/// of two integer or floating-point types is the integer or floating-point type with the smallest
/// size that is larger than or equal to that of both arguments. For instance, the least upper
/// bound of an int16 and an int32 is int32.
fn lub_type(l: Type, r: Type, i: &Info) -> PyResult<Type> {
    match (l.get_scalar_elem_size(), r.get_scalar_elem_size()) {
        (Some(lsz), Some(rsz)) => {
            Ok(Type::Tensor {sz: lub_elem_size(lsz, rsz, i)?, shape: vec![]})
        },
        (None, None) if l.eq(&r) => Ok(l),
        _ => py_type_error!(i, "Cannot unify incompatible types {l} and {r}"),
    }
}

fn type_check_builtin(
    func: Builtin,
    mut args: Vec<Expr>,
    i: Info
) -> PyResult<Expr> {
    match &func {
        // Literals
        Builtin::Inf if args.is_empty() => {
            Ok(Expr::Builtin {func, args, ty: Type::Tensor {sz: ElemSize::F64, shape: vec![]}, i})
        },
        // Unary operations on (floating-point) scalar values
        Builtin::Exp | Builtin::Log | Builtin::Cos | Builtin::Sin |
        Builtin::Sqrt if args.len() == 1 => {
            let ty = args[0].get_type().clone();
            if ty.is_floating_point() {
                Ok(Expr::Builtin {func, args, ty, i})
            } else {
                py_type_error!(i, "Unexpected type {ty} of unary builtin (expected float)")
            }
        },
        // CUDA has no built-in tanh for 16-bit floats.
        Builtin::Tanh if args.len() == 1 => {
            let ty = args[0].get_type().clone();
            match ty.get_scalar_elem_size() {
                Some(ElemSize::F16) =>
                    py_type_error!(i, "Operation tanh not supporteed for 16-bit floats"),
                Some(ElemSize::F32 | ElemSize::F64) => Ok(Expr::Builtin {func, args, ty, i}),
                _ => py_type_error!(i, "Unexpected type {ty} of tanh builtin (expected float)")
            }
        },
        Builtin::Abs if args.len() == 1 => {
            let ty = args[0].get_type().clone();
            if ty.is_signed_integer() || ty.is_floating_point() {
                Ok(Expr::Builtin {func, args, ty, i})
            } else {
                py_type_error!(i, "Unexpected type {ty} of abs builtin")
            }
        },
        // Unary cast operation on scalar values
        Builtin::Convert {sz} if args.len() == 1 => {
            let arg = args.remove(0);
            let ty = arg.get_type().clone();
            if ty.get_scalar_elem_size().is_some() {
                Ok(Expr::Convert {
                    e: Box::new(arg),
                    ty: Type::Tensor {sz: sz.clone(), shape: vec![]}
                })
            } else {
                py_type_error!(i, "Unexpected type {ty} of type conversion")
            }
        },
        // Binary operations on scalar values
        Builtin::Max | Builtin::Min | Builtin::Atan2 if args.len() == 2 => {
            let mk_builtin = |func, fst, snd, ty, i| {
                let fst = coerce_type(fst, &ty)?;
                let snd = coerce_type(snd, &ty)?;
                let args = vec![fst, snd];
                Ok(Expr::Builtin {func, args, ty, i})
            };
            let snd = args.pop().unwrap();
            let fst = args.pop().unwrap();
            let ty = lub_type(fst.get_type().clone(), snd.get_type().clone(), &i)?;
            match ty.get_scalar_elem_size() {
                Some(_) if func != Builtin::Atan2 => mk_builtin(func, fst, snd, ty, i),
                Some(ElemSize::F64) if func == Builtin::Atan2 => {
                    mk_builtin(func, fst, snd, ty, i)
                },
                _ => {
                    py_type_error!(i, "Unexpected type {ty} of binary builtin")
                }
            }
        },
        _ => py_type_error!(i, "Unsupported use of builtin {func}")
    }
}

fn type_check_unop(
    op: &UnOp,
    arg: &Expr,
    i: &Info
) -> PyResult<Type> {
    let ty = arg.get_type();
    match op {
        UnOp::Sub => {
            if ty.is_signed_integer() || ty.is_floating_point() {
                Ok(ty.clone())
            } else {
                py_type_error!(i, "Invalid type {ty} of unary minus")
            }
        },
        UnOp::Not => {
            if ty.is_boolean() {
                Ok(ty.clone())
            } else {
                py_type_error!(i, "Invalid type {ty} of boolean negation")
            }
        }
        UnOp::BitNeg => {
            if ty.is_signed_integer() {
                Ok(ty.clone())
            } else {
                py_type_error!(i, "Invalid type {ty} of bitwise negation")
            }
        },
    }
}

fn type_check_binop(
    lhs: Expr,
    op: &BinOp,
    rhs: Expr,
    i: &Info
) -> PyResult<(Box<Expr>, Type, Box<Expr>)> {
    let lty = lhs.get_type().clone();
    let rty = rhs.get_type().clone();
    let ty = lub_type(lty, rty, i)?;
    let lhs = coerce_type(lhs, &ty)?;
    let rhs = coerce_type(rhs, &ty)?;
    let ty = match op {
        // Arithmetic operations supporting either integers or floating point numbers
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
            if ty.is_signed_integer() || ty.is_floating_point() {
                Ok(ty)
            } else {
                py_type_error!(i, "Invalid type {ty} of arithmetic operation")
            }
        },
        // Arithmetic operations only supported for integers
        BinOp::FloorDiv | BinOp::Mod => {
            if ty.is_signed_integer() {
                Ok(ty)
            } else {
                py_type_error!(i, "Invalid type {ty} of integer arithmetic operation")
            }
        },
        // Arithmetic operations only supported for floating-point numbers
        BinOp::Pow => {
            match ty.get_scalar_elem_size() {
                Some(ElemSize::F32 | ElemSize::F64) => Ok(ty),
                _ => py_type_error!(i, "Invalid type {ty} of floating-poont arithmetic operation")
            }
        },
        // Boolean operations
        BinOp::And | BinOp::Or => {
            if ty.is_boolean() {
                Ok(ty)
            } else {
                py_type_error!(i, "Invalid type {ty} of boolean operation")
            }
        },
        // Bitwise operations
        BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::BitShl | BinOp::BitShr => {
            if ty.is_signed_integer() {
                Ok(ty)
            } else {
                py_type_error!(i, "Invalid type {ty} of bitwise operation")
            }
        },
        // Boolean comparison operations, allowing comparison between elementary types
        BinOp::Eq | BinOp::Neq | BinOp::Leq | BinOp::Geq | BinOp::Lt | BinOp::Gt => {
            if let Some(_) = ty.get_scalar_elem_size() {
                Ok(Type::Tensor {sz: ElemSize::Bool, shape: vec![]})
            } else {
                py_type_error!(i, "Invalid type {ty} of boolean comparison operation")
            }
        },
    }?;
    Ok((Box::new(lhs), ty, Box::new(rhs)))
}

fn type_check_expr(
    vars: &BTreeMap<Name, Type>,
    e: Expr
) -> PyResult<Expr> {
    match e {
        Expr::Var {id, i, ..} => {
            let ty = match vars.get(&id) {
                Some(ty) if ty != &Type::Unknown => Ok(ty.clone()),
                _ => py_type_error!(i, "Variable {id} has unknown type")
            }?;
            Ok(Expr::Var {id, ty, i})
        },
        Expr::String {v, i, ..} => Ok(Expr::String {v, ty: Type::String, i}),
        Expr::Bool {v, i, ..} => {
            let ty = Type::Tensor {sz: ElemSize::Bool, shape: vec![]};
            Ok(Expr::Bool {v, ty, i})
        },
        Expr::Int {v, i, ..} =>
            Ok(Expr::Int {v, ty: Type::Tensor {sz: ElemSize::I64, shape: vec![]}, i}),
        Expr::Float {v, i, ..} =>
            Ok(Expr::Float {v, ty: Type::Tensor {sz: ElemSize::F64, shape: vec![]}, i}),
        Expr::UnOp {op, arg, i, ..} => {
            let arg = Box::new(type_check_expr(vars, *arg)?);
            let ty = type_check_unop(&op, &arg, &i)?;
            Ok(Expr::UnOp {op, arg, ty, i})
        },
        Expr::BinOp {lhs, op, rhs, i, ..} => {
            let lhs = type_check_expr(vars, *lhs)?;
            let rhs = type_check_expr(vars, *rhs)?;
            let (lhs, ty, rhs) = type_check_binop(lhs, &op, rhs, &i)?;
            Ok(Expr::BinOp {lhs, op, rhs, ty, i})
        },
        Expr::IfExpr {cond, thn, els, i, ..} => {
            let cond = Box::new(type_check_expr(vars, *cond)?);
            let ty = cond.get_type();
            if ty.is_boolean() {
                let thn = type_check_expr(vars, *thn)?;
                let els = type_check_expr(vars, *els)?;
                let thn_ty = thn.get_type().clone();
                let els_ty = els.get_type().clone();
                let ty = lub_type(thn_ty, els_ty, &i)?;
                let thn = Box::new(coerce_type(thn, &ty)?);
                let els = Box::new(coerce_type(els, &ty)?);
                Ok(Expr::IfExpr {cond, thn, els, ty, i})
            } else {
                py_type_error!(i, "If expression has condition of invalid type {ty}")
            }
        },
        Expr::Subscript {target, idx, i, ..} => {
            let target = type_check_expr(vars, *target)?;
            let (ty, idx) = match *idx {
                Expr::String {v, i, ..} => {
                    if let Type::Dict {fields} = target.get_type() {
                        if let Some(ty) = fields.get(&v) {
                            let idx_ty = Type::String;
                            Ok((ty.clone(), Expr::String {v, ty: idx_ty, i}))
                        } else {
                            py_type_error!(i, "Field {v} not present in {0}", target.get_type())
                        }
                    } else {
                        py_type_error!(i, "Cannot index using a string on non-dict expression")
                    }
                },
                idx => {
                    let idx = type_check_expr(vars, idx)?;
                    let elem_ty = if let Type::Tensor {sz, shape} = target.get_type() {
                        let idx_dims = match idx.get_type() {
                            Type::Tensor {shape, ..} if shape.len() == 0 => Ok(1),
                            Type::Tuple {elems} => Ok(elems.len()),
                            ty => py_type_error!(i, "Unsupported index of type {ty}")
                        }?;
                        if idx_dims <= shape.len() {
                            let res_shape = shape.clone()
                                .into_iter()
                                .skip(idx_dims)
                                .collect::<Vec<i64>>();
                            Ok(Type::Tensor {sz: sz.clone(), shape: res_shape})
                        } else {
                            let sh = shape.iter().map(|i| i.to_string()).join(",");
                            py_type_error!(i, "Indexing with {idx_dims} dimensions on tensor of shape [{sh}]")
                        }
                    } else {
                        py_type_error!(i, "Subscript operation on unsupported target {target}")
                    }?;
                    match idx.get_type() {
                        Type::Tensor {shape, ..} if shape.len() == 0 => {
                            let expected_ty = Type::Tensor {sz: ElemSize::I64, shape: vec![]};
                            Ok((elem_ty, coerce_type(idx, &expected_ty)?))
                        },
                        Type::Tuple {elems} => {
                            let expected_types = elems.iter()
                                .map(|_| Type::Tensor {sz: ElemSize::I64, shape: vec![]})
                                .collect::<Vec<Type>>();
                            let expected_ty = Type::Tuple {elems: expected_types};
                            Ok((elem_ty, coerce_type(idx, &expected_ty)?))
                        },
                        ty => py_type_error!(i, "Unsupported index of type {ty} in subscript operation")
                    }
                }
            }?;
            Ok(Expr::Subscript {target: Box::new(target), idx: Box::new(idx), ty, i})
        },
        Expr::Tuple {elems, i, ..} => {
            let elems = elems.into_iter()
                .map(|e| type_check_expr(vars, e))
                .collect::<PyResult<Vec<Expr>>>()?;
            let elem_types = elems.iter()
                .map(|e| e.get_type().clone())
                .collect::<Vec<Type>>();
            let ty = Type::Tuple {elems: elem_types};
            Ok(Expr::Tuple {elems, ty, i})
        },
        Expr::Dict {fields, i, ..} => {
            let fields = fields.into_iter()
                .map(|(k, v)| Ok((k, type_check_expr(vars, v)?)))
                .collect::<PyResult<BTreeMap<String, Expr>>>()?;
            let ty_fields = fields.iter()
                .map(|(k, v)| (k.clone(), v.get_type().clone()))
                .collect::<BTreeMap<String, Type>>();
            let ty = Type::Dict {fields: ty_fields};
            Ok(Expr::Dict {fields, ty, i})
        },
        Expr::Builtin {func, args, i, ..} => {
            let args = type_check_exprs(vars, args)?;
            type_check_builtin(func, args, i)
        },
        e @ Expr::Convert {..} => Ok(e)
    }
}

fn type_check_exprs(
    vars: &BTreeMap<Name, Type>,
    exprs: Vec<Expr>
) -> PyResult<Vec<Expr>> {
    exprs.into_iter()
        .map(|e| type_check_expr(vars, e))
        .collect()
}

fn validate_condition_type(cond: Expr, i: &Info) -> PyResult<Expr> {
    let ty = cond.get_type();
    match ty {
        Type::Tensor {..} => Ok(cond),
        _ => py_type_error!(i, "Unsupported type {ty} of conditional expression")
    }
}

fn type_check_stmt(
    mut vars: BTreeMap<Name, Type>,
    stmt: Stmt
) -> PyResult<(BTreeMap<Name, Type>, Stmt)> {
    match stmt {
        Stmt::Definition {id, expr, i, ..} => {
            let expr = type_check_expr(&vars, expr)?;
            let ty = expr.get_type().clone();
            vars.insert(id.clone(), ty.clone());
            Ok((vars, Stmt::Definition {ty, id, expr, i}))
        },
        Stmt::Assign {dst, expr, i} => {
            let dst = type_check_expr(&vars, dst)?;
            let expr = type_check_expr(&vars, expr)?;
            let expr = coerce_type(expr, dst.get_type())?;
            Ok((vars, Stmt::Assign {dst, expr, i}))
        },
        Stmt::For {var, lo, hi, step, body, i} => {
            let lo = type_check_expr(&vars, lo)?;
            let lo = ensure_scalar_type(lo, ElemSize::I64)?;
            let hi = type_check_expr(&vars, hi)?;
            let hi = ensure_scalar_type(hi, ElemSize::I64)?;
            let mut body_vars = vars.clone();
            body_vars.insert(var.clone(), Type::Tensor {sz: ElemSize::I64, shape: vec![]});
            let (_, body) = type_check_stmts(body_vars, body)?;
            Ok((vars, Stmt::For {var, lo, hi, step, body, i}))
        },
        Stmt::If {cond, thn, els, i} => {
            let cond = validate_condition_type(type_check_expr(&vars, cond)?, &i)?;
            let (_, thn) = type_check_stmts(vars.clone(), thn)?;
            let (_, els) = type_check_stmts(vars.clone(), els)?;
            Ok((vars, Stmt::If {cond, thn, els, i}))
        },
        Stmt::While {cond, body, i} => {
            let cond = validate_condition_type(type_check_expr(&vars, cond)?, &i)?;
            let (_, body) = type_check_stmts(vars.clone(), body)?;
            Ok((vars, Stmt::While {cond, body, i}))
        },
        Stmt::WithGpuContext {body, i} => {
            let (_, body) = type_check_stmts(vars.clone(), body)?;
            Ok((vars, Stmt::WithGpuContext {body, i}))
        },
        Stmt::Label {label, assoc, i} => {
            let (vars, assoc) = match assoc {
                Some(s) => {
                    let (vars, s) = type_check_stmt(vars, *s)?;
                    (vars, Some(Box::new(s)))
                },
                None => (vars, None)
            };
            Ok((vars, Stmt::Label {label, assoc, i}))
        }
    }
}

fn type_check_stmts(
    vars: BTreeMap<Name, Type>,
    stmts: Vec<Stmt>
) -> PyResult<(BTreeMap<Name, Type>, Vec<Stmt>)> {
    stmts.into_iter()
        .fold(Ok((vars, vec![])), |acc: PyResult<_>, stmt| {
            let (vars, mut stmts) = acc?;
            let (vars, stmt) = type_check_stmt(vars, stmt)?;
            stmts.push(stmt);
            Ok((vars, stmts))
        })
}

fn type_check_body(
    body: Vec<Stmt>,
    params: Vec<Param>
) -> PyResult<Vec<Stmt>> {
    let vars = params.iter()
        .map(|Param {id, ty, ..}| (id.clone(), ty.clone()))
        .collect::<BTreeMap<Name, Type>>();
    let (_, body) = type_check_stmts(vars, body)?;
    Ok(body)
}

pub fn type_check<'py>(
    def: FunDef,
    args: &Vec<Bound<'py, PyAny>>
) -> PyResult<FunDef> {
    let FunDef {id, params, body, i} = def;
    let params = add_param_types(&id, params, args)?;
    let body = type_check_body(body, params.clone())?;
    Ok(FunDef {id, params, body, i})
}

#[cfg(test)]
mod test {
    use super::*;

    use strum::IntoEnumIterator;

    fn test_lub_elem_size_ok(lhs: &ElemSize, rhs: &ElemSize, expected: ElemSize) {
        let result = lub_elem_size(lhs, rhs, &Info::default());
        assert_eq!(expected, result.unwrap());
    }

    fn test_lub_elem_size_fail(lhs: &ElemSize, rhs: &ElemSize) {
        let result = lub_elem_size(lhs, rhs, &Info::default());
        assert!(result.is_err());
    }

    fn scalar_type(sz: ElemSize) -> Type {
        Type::Tensor {sz, shape: vec![]}
    }

    fn bool_type() -> Type {
        scalar_type(ElemSize::Bool)
    }

    #[test]
    fn lub_elem_size_equals() {
        for sz in ElemSize::iter() {
            test_lub_elem_size_ok(&sz, &sz, sz.clone())
        }
    }

    #[test]
    fn lub_elem_size_is_commutative() {
        for sz1 in ElemSize::iter() {
            for sz2 in ElemSize::iter() {
                let i = Info::default();
                let r1 = lub_elem_size(&sz1, &sz2, &i);
                let r2 = lub_elem_size(&sz2, &sz1, &i);
                if r1.is_ok() && r2.is_ok() {
                    assert_eq!(r1.unwrap(), r2.unwrap())
                } else if r1.is_err() ^ r2.is_err() {
                    assert!(false)
                }
            }
        }
    }

    #[test]
    fn lub_elem_size_f32_f64() {
        test_lub_elem_size_ok(&ElemSize::F32, &ElemSize::F64, ElemSize::F64)
    }

    #[test]
    fn lub_elem_size_int_float() {
        test_lub_elem_size_fail(&ElemSize::I32, &ElemSize::F32)
    }

    fn test_lub_type_ok(lty: Type, rty: Type, expected: Type) {
        let r = lub_type(lty, rty, &Info::default());
        assert_eq!(expected, r.unwrap());
    }

    fn test_lub_type_fail(lty: Type, rty: Type) {
        let r = lub_type(lty, rty, &Info::default());
        assert!(r.is_err());
    }
    
    #[test]
    fn lub_type_string() {
        test_lub_type_ok(Type::String, Type::String, Type::String)
    }

    #[test]
    fn lub_type_elem_eq() {
        let ty = scalar_type(ElemSize::I16);
        test_lub_type_ok(ty.clone(), ty.clone(), ty.clone())
    }

    #[test]
    fn lub_type_elem_compatible() {
        let ty1 = scalar_type(ElemSize::F32);
        let ty2 = scalar_type(ElemSize::F64);
        test_lub_type_ok(ty1.clone(), ty2.clone(), ty2.clone())
    }

    #[test]
    fn lub_type_elem_incompatible() {
        let ty1 = scalar_type(ElemSize::F32);
        let ty2 = scalar_type(ElemSize::I8);
        test_lub_type_fail(ty1, ty2)
    }

    #[test]
    fn lub_type_bool_eq() {
        let ty = scalar_type(ElemSize::Bool);
        test_lub_type_ok(ty.clone(), ty.clone(), ty.clone())
    }

    #[test]
    fn lub_type_tensor_equal_ok() {
        let ty = Type::Tensor {sz: ElemSize::I32, shape: vec![5]};
        test_lub_type_ok(ty.clone(), ty.clone(), ty.clone())
    }

    #[test]
    fn lub_type_tensor_compatible_fails() {
        let ty1 = Type::Tensor {sz: ElemSize::F32, shape: vec![5]};
        let ty2 = Type::Tensor {sz: ElemSize::F64, shape: vec![5]};
        test_lub_type_fail(ty1, ty2)
    }

    #[test]
    fn lub_type_tensor_different_shape_fails() {
        let ty1 = Type::Tensor {sz: ElemSize::F32, shape: vec![5]};
        let ty2 = Type::Tensor {sz: ElemSize::F32, shape: vec![4]};
        test_lub_type_fail(ty1, ty2)
    }

    #[test]
    fn lub_type_tuple_eq_elems() {
        let ty = Type::Tuple {elems: vec![
            bool_type(),
            scalar_type(ElemSize::F32)
        ]};
        test_lub_type_ok(ty.clone(), ty.clone(), ty.clone())
    }

    #[test]
    fn lub_type_tuple_compatible_elems_fails() {
        let ty1 = Type::Tuple {elems: vec![scalar_type(ElemSize::F32)]};
        let ty2 = Type::Tuple {elems: vec![scalar_type(ElemSize::F64)]};
        test_lub_type_fail(ty1, ty2)
    }

    #[test]
    fn lub_type_dict() {
        let ty = Type::Dict {fields: BTreeMap::new()};
        test_lub_type_ok(ty.clone(), ty.clone(), ty.clone())
    }

    fn var(s: &str) -> Name {
        Name::new(s.to_string())
    }

    fn test_tc_unop(op: UnOp, arg: Expr) -> PyResult<Type> {
        type_check_unop(&op, &arg, &Info::default())
    }

    #[test]
    fn type_check_unop_signed_int_negation() {
        let ty = scalar_type(ElemSize::I64);
        let arg = Expr::Int {v: 1, ty: ty.clone(), i: Info::default()};
        let res = test_tc_unop(UnOp::Sub, arg).unwrap();
        assert_eq!(res, ty);
    }

    #[test]
    fn type_check_unop_float_negation() {
        let ty = scalar_type(ElemSize::F32);
        let arg = Expr::Var {id: var("x"), ty: ty.clone(), i: Info::default()};
        let res = test_tc_unop(UnOp::Sub, arg).unwrap();
        assert_eq!(res, ty);
    }

    fn test_tc_binop(lhs: Expr, op: BinOp, rhs: Expr) -> PyResult<Type> {
        let (_, ty, _) = type_check_binop(lhs, &op, rhs, &Info::default())?;
        Ok(ty)
    }

    #[test]
    fn type_check_binop_signed_int_addition() {
        let ty = scalar_type(ElemSize::I64);
        let lhs = Expr::Int {v: 1, ty: ty.clone(), i: Info::default()};
        let rhs = Expr::Int {v: 2, ty: ty.clone(), i: Info::default()};
        let res = test_tc_binop(lhs, BinOp::Add, rhs).unwrap();
        assert_eq!(res, ty);
    }

    #[test]
    fn type_check_binop_coerced_signed_int_multiplication() {
        let lty = scalar_type(ElemSize::I32);
        let lhs = Expr::Var {id: var("x"), ty: lty.clone(), i: Info::default()};
        let rty = scalar_type(ElemSize::I16);
        let rhs = Expr::Var {id: var("y"), ty: rty, i: Info::default()};
        let res = test_tc_binop(lhs, BinOp::Mul, rhs).unwrap();
        assert_eq!(res, lty);
    }

    #[test]
    fn type_check_binop_float_subtraction() {
        let ty = scalar_type(ElemSize::F32);
        let lhs = Expr::Float {v: 3.14, ty: ty.clone(), i: Info::default()};
        let rhs = Expr::Var {id: var("x"), ty: ty.clone(), i: Info::default()};
        let res = test_tc_binop(lhs, BinOp::Sub, rhs).unwrap();
        assert_eq!(res, ty);
    }

    #[test]
    fn type_check_int_equality() {
        let ty = scalar_type(ElemSize::I16);
        let lhs = Expr::Int {v: 1, ty: ty.clone(), i: Info::default()};
        let rhs = Expr::Int {v: 2, ty: ty.clone(), i: Info::default()};
        let res = test_tc_binop(lhs, BinOp::Eq, rhs).unwrap();
        assert_eq!(res, bool_type());
    }

    #[test]
    fn type_check_float_lt() {
        let ty = scalar_type(ElemSize::F32);
        let lhs = Expr::Float {v: 2.718, ty: ty.clone(), i: Info::default()};
        let rhs = Expr::Var {id: var("x"), ty: ty.clone(), i: Info::default()};
        let res = test_tc_binop(lhs, BinOp::Lt, rhs).unwrap();
        assert_eq!(res, bool_type());
    }

    fn make_map<'a>(entries: Vec<(&'a str, Type)>) -> BTreeMap<Name, Type> {
        entries.into_iter()
            .map(|(id, ty)| (Name::new(id.to_string()), ty))
            .collect::<BTreeMap<Name, Type>>()
    }

    #[test]
    fn type_check_expr_known_var() {
        let vars = make_map(vec![("x", bool_type())]);
        let v = Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()};
        let r = type_check_expr(&vars, v);
        assert!(r.is_ok());
        assert_eq!(r.unwrap().get_type().clone(), bool_type());
    }

    #[test]
    fn type_check_expr_unknown_var() {
        let vars = make_map(vec![]);
        let v = Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()};
        assert!(type_check_expr(&vars, v).is_err())
    }

    #[test]
    fn type_check_expr_string_literal() {
        let vars = make_map(vec![]);
        let v = Expr::String {v: "x".to_string(), ty: Type::Unknown, i: Info::default()};
        let r = type_check_expr(&vars, v);
        assert!(r.is_ok());
        assert_eq!(r.unwrap().get_type().clone(), Type::String);
    }

    #[test]
    fn type_check_expr_int_literal() {
        let vars = make_map(vec![]);
        let v = Expr::Int {v: 0, ty: Type::Unknown, i: Info::default()};
        let r = type_check_expr(&vars, v);
        assert!(r.is_ok());
        assert_eq!(r.unwrap().get_type().clone(), scalar_type(ElemSize::I64));
    }

    #[test]
    fn type_check_expr_float_literal() {
        let vars = make_map(vec![]);
        let v = Expr::Float {v: 0.0, ty: Type::Unknown, i: Info::default()};
        let r = type_check_expr(&vars, v);
        assert!(r.is_ok());
        assert_eq!(r.unwrap().get_type().clone(), scalar_type(ElemSize::F64));
    }

    #[test]
    fn type_check_expr_dict_lookup() {
        let fields = vec![("a", bool_type())].into_iter()
            .map(|(id, ty)| (id.to_string(), ty))
            .collect::<BTreeMap<String, Type>>();
        let dict_ty = Type::Dict {fields};
        let vars = make_map(vec![("x", dict_ty.clone())]);
        let v = Expr::Subscript {
            target: Box::new(Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()}),
            idx: Box::new(Expr::String {v: "a".to_string(), ty: Type::Unknown, i: Info::default()}),
            ty: Type::Unknown,
            i: Info::default()
        };
        let r = type_check_expr(&vars, v);
        assert!(r.is_ok());
        if let Expr::Subscript {target, idx, ty, ..} = r.unwrap() {
            assert_eq!(ty, bool_type());
            assert_eq!(target.get_type().clone(), dict_ty);
            assert_eq!(idx.get_type().clone(), Type::String);
        } else {
            assert!(false);
        }
    }

    #[test]
    fn type_check_expr_tensor_lookup() {
        let tensor_ty = Type::Tensor {sz: ElemSize::F32, shape: vec![5]};
        let vars = make_map(vec![("x", tensor_ty.clone())]);
        let v = Expr::Subscript {
            target: Box::new(Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()}),
            idx: Box::new(Expr::Int {v: 0, ty: Type::Unknown, i: Info::default()}),
            ty: Type::Unknown,
            i: Info::default()
        };
        let r = type_check_expr(&vars, v);
        assert!(r.is_ok());
        if let Expr::Subscript {target, idx, ty, ..} = r.unwrap() {
            assert_eq!(ty, scalar_type(ElemSize::F32));
            assert_eq!(target.get_type().clone(), tensor_ty.clone());
            assert_eq!(idx.get_type().clone(), scalar_type(ElemSize::I64));
        } else {
            assert!(false);
        }
    }

    #[test]
    fn type_check_expr_tensor_lookup_with_conversion() {
        let tensor_ty = Type::Tensor {sz: ElemSize::F32, shape: vec![5]};
        let vars = make_map(vec![
            ("x", tensor_ty.clone()),
            ("y", scalar_type(ElemSize::I32))
        ]);
        let v = Expr::Subscript {
            target: Box::new(Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()}),
            idx: Box::new(Expr::Var {id: var("y"), ty: Type::Unknown, i: Info::default()}),
            ty: Type::Unknown,
            i: Info::default()
        };
        let r = type_check_expr(&vars, v);
        if let Expr::Subscript {target, idx, ty, ..} = r.unwrap() {
            assert_eq!(ty, scalar_type(ElemSize::F32));
            assert_eq!(target.get_type().clone(), tensor_ty);
            // As the variable y is a 32-bit integer, the type-checker should insert a Convert node
            // indicating that it needs to be converted to a 64-bit signed integer value (we always
            // expect this for indexing operations).
            if let Expr::Convert {e, ty} = *idx {
                assert_eq!(e.get_type().clone(), scalar_type(ElemSize::I32));
                assert_eq!(ty, scalar_type(ElemSize::I64));
            } else {
                assert!(false);
            }
        } else {
            assert!(false);
        }
    }

    #[test]
    fn type_check_expr_tensor_slicing() {
        let tensor_ty = Type::Tensor {sz: ElemSize::F32, shape: vec![5,6,4]};
        let vars = make_map(vec![
            ("x", tensor_ty.clone()),
        ]);
        let tuple_ty = Type::Tuple {
            elems: vec![scalar_type(ElemSize::I64), scalar_type(ElemSize::I64)]
        };
        let idx = Box::new(Expr::Tuple {
            elems: vec![
                Expr::Int {v: 2, ty: scalar_type(ElemSize::I64), i: Info::default()},
                Expr::Int {v: 5, ty: scalar_type(ElemSize::I64), i: Info::default()}
            ],
            ty: tuple_ty.clone(),
            i: Info::default()
        });
        let v = Expr::Subscript {
            target: Box::new(Expr::Var {id: var("x"), ty: Type::Unknown, i: Info::default()}),
            idx, ty: Type::Unknown, i: Info::default()
        };
        let r = type_check_expr(&vars, v);
        if let Expr::Subscript {target, idx, ty, ..} = r.unwrap() {
            assert_eq!(ty, Type::Tensor {sz: ElemSize::F32, shape: vec![4]});
            assert_eq!(target.get_type().clone(), tensor_ty);
            assert_eq!(idx.get_type().clone(), tuple_ty);
        } else {
            assert!(false);
        }
    }
}
