use super::par::{GpuMap, GpuMapping};
use crate::parir_compile_error;
use crate::ir::ast::*;
use crate::utils::err::*;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use std::collections::{BTreeMap, BTreeSet};

fn collect_sync_points_stmt(
    acc: CompileResult<BTreeSet<Name>>,
    stmt: &Stmt
) -> CompileResult<BTreeSet<Name>> {
    match stmt {
        Stmt::For {var, body, par, ..} => {
            let mut sync = acc?;
            if par.is_parallel() {
                sync.insert(var.clone());
            };
            collect_sync_points_stmts(Ok(sync), body)
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::While {..} |
        Stmt::If {..} => {
            stmt.sfold(acc, collect_sync_points_stmt)
        }
    }
}

fn collect_sync_points_stmts(
    acc: CompileResult<BTreeSet<Name>>,
    stmts: &Vec<Stmt>
) -> CompileResult<BTreeSet<Name>> {
    stmts.iter().fold(acc, collect_sync_points_stmt)
}

fn remove_redundant_sync_par_stmt(
    sync: BTreeSet<Name>,
    stmt: &Stmt
) -> BTreeSet<Name> {
    match stmt {
        Stmt::For {var, body, ..} => {
            let is_par = sync.contains(var);
            remove_redundant_sync_par_stmts(sync, body, is_par)
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::While {..} |
        Stmt::If {..} => {
            stmt.sfold(sync, remove_redundant_sync_par_stmt)
        },
    }
}

fn remove_redundant_sync_par_stmts(
    mut sync: BTreeSet<Name>,
    stmts: &Vec<Stmt>,
    in_par: bool
) -> BTreeSet<Name> {
    let sync = match stmts.last() {
        Some(Stmt::For {var, body, par, ..}) => {
            if in_par && par.is_parallel() && !par.reduction {
                sync.remove(&var);
            };
            remove_redundant_sync_par_stmts(sync, &body, par.is_parallel())
        },
        Some(s @ (Stmt::Definition {..} | Stmt::Assign {..} | Stmt::While {..} |
                  Stmt::If {..})) => {
            s.sfold(sync, remove_redundant_sync_par_stmt)
        },
        None => sync,
    };
    stmts.iter()
        .rev()
        .skip(1)
        .fold(sync, remove_redundant_sync_par_stmt)
}

fn remove_redundant_sync_stmt(
    mut sync: BTreeSet<Name>,
    stmt: &Stmt
) -> BTreeSet<Name> {
    match stmt {
        Stmt::For {var, body, ..} => {
            let is_par = sync.contains(var);
            if is_par {
                sync.remove(var);
                remove_redundant_sync_par_stmts(sync, body, true)
            } else {
                remove_redundant_sync_stmts(sync, body)
            }
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::While {..} |
        Stmt::If {..} => {
            stmt.sfold(sync, remove_redundant_sync_stmt)
        }
    }
}

fn remove_redundant_sync_stmts(
    sync: BTreeSet<Name>,
    stmts: &Vec<Stmt>,
) -> BTreeSet<Name> {
    stmts.sfold(sync, remove_redundant_sync_stmt)
}

fn ensure_no_inter_block_sync_par_stmt(
    stmt: &Stmt,
    sync: &BTreeSet<Name>,
    pars: &[GpuMap]
) -> CompileResult<()> {
    match stmt {
        Stmt::For {var, body, par, i, ..} => {
            let pars = if sync.contains(var) {
                match &pars[0] {
                    GpuMap::Thread {..} => {
                        Ok(())
                    },
                    GpuMap::Block {..} | GpuMap::ThreadBlock {..} => {
                        let msg = if par.reduction {
                            concat!(
                                "Parallel reductions using more than 1024 ",
                                "threads require inter-block synchronization, ",
                                "which is not supported.")
                        } else {
                            concat!(
                                "This parallel for-loop uses more than 1024 ",
                                "threads and therefore requires inter-block ",
                                "synchronization, which is not supported.")
                        };
                        parir_compile_error!(i, "{}", msg)
                    },
                }?;
                &pars[1..]
            } else if par.is_parallel() {
                &pars[1..]
            } else {
                pars
            };
            ensure_no_inter_block_sync_par_stmts(body, sync, pars)
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::While {..} |
        Stmt::If {..} => {
            stmt.sfold_result(Ok(()), |_, s| {
                ensure_no_inter_block_sync_par_stmt(s, sync, pars)
            })
        }
    }
}

fn ensure_no_inter_block_sync_par_stmts(
    stmts: &Vec<Stmt>,
    sync: &BTreeSet<Name>,
    pars: &[GpuMap]
) -> CompileResult<()> {
    stmts.sfold_result(Ok(()), |_, s| {
        ensure_no_inter_block_sync_par_stmt(s, sync, pars)
    })
}

fn ensure_no_inter_block_sync_stmt(
    stmt: &Stmt,
    sync: &BTreeSet<Name>,
    gpu_mapping: &BTreeMap<Name, GpuMapping>
) -> CompileResult<()> {
    match stmt {
        Stmt::For {var, body, ..} => {
            match gpu_mapping.get(var) {
                Some(m) => {
                    let map = &m.get_mapping()[1..];
                    ensure_no_inter_block_sync_par_stmts(body, sync, map)
                },
                None => ensure_no_inter_block_sync_stmts(body, sync, gpu_mapping)
            }
        },
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::While {..} |
        Stmt::If {..} => {
            stmt.sfold_result(Ok(()), |_, s| {
                ensure_no_inter_block_sync_stmt(s, sync, gpu_mapping)
            })
        }
    }
}

fn ensure_no_inter_block_sync_stmts(
    stmts: &Vec<Stmt>,
    sync: &BTreeSet<Name>,
    gpu_mapping: &BTreeMap<Name, GpuMapping>
) -> CompileResult<()> {
    stmts.sfold_result(Ok(()), |_, s| {
        ensure_no_inter_block_sync_stmt(s, sync, gpu_mapping)
    })
}

/// Identify where we need to insert synchronization points in the AST (after which parallel
/// for-loops, identified by the name of the iteration variable). In this system, every parallel
/// for-loop has an implicit synchronization point after it. That is, we consider the iterations of
/// a parallel loop to run in an arbitrary order, but they all need to complete before executing
/// later statements.
///
/// The end of a kernel is an implicit synchronization point, because the CUDA model guarantees
/// that one kernel completes before the next one starts executing. Therefore, we do not need to
/// synchronize the outermost parallel for-loop. Further, if the final statement within a parallel
/// for-loop is another parallelized for-loop, we do not need to synchronize it, as the iterations
/// of the outer for-loop are assumed to be independent.
///
/// Finally, the only general way to achieve synchronization across CUDA blocks is to split up code
/// into separate kernels. The current implementation does not support this kind of transformation.
/// Therefore, synchronization points are only allowed when it involves the threads of a single
/// block, because in this case we can synchronize using a CUDA intrinsic.
pub fn identify_sync_points(
    ast: &Ast,
    gpu_mapping: &BTreeMap<Name, GpuMapping>
) -> CompileResult<BTreeSet<Name>> {
    // Collect a synchronization point for the end of each parallel for-loop.
    let sync = collect_sync_points_stmts(Ok(BTreeSet::new()), &ast.fun.body)?;

    // Remove synchronization points for for-loops that run as the final statement of a parallel
    // for-loop, and for the outermost parallel for-loop. The iterations of a parallel for-loop can
    // execute in any order, so synchronizing at the end of an iteration is redundant. Also, the
    // outermost parallel for-loop becomes the entry point to the CUDA kernel, and CUDA
    // automatically performs synchronization between kernels, so this synchronization point can
    // also be omitted.
    let sync = remove_redundant_sync_stmts(sync, &ast.fun.body);

    // Ensure that the remaining synchronization points are executed within a block only. This is
    // important because CUDA has no good way to perform this. The best option is to split up code
    // into multiple kernels, but this is a non-trivial problem.
    ensure_no_inter_block_sync_stmts(&ast.fun.body, &sync, gpu_mapping)?;

    Ok(sync)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ir_builder::*;
    use crate::cuda::ast::{Dim, LaunchArgs};
    use crate::cuda::par::DEFAULT_TPB;

    fn make_mapping(map: Vec<(Name, GpuMapping)>) -> BTreeMap<Name, GpuMapping> {
        map.into_iter().collect::<_>()
    }

    fn assert_sync(ast: Ast, mapping: BTreeMap<Name, GpuMapping>, expected: BTreeSet<Name>) {
        let sync = collect_sync_points_stmts(Ok(BTreeSet::new()), &ast.fun.body).unwrap();
        let sync = remove_redundant_sync_stmts(sync, &ast.fun.body);
        assert_eq!(sync, expected);
        ensure_no_inter_block_sync_stmts(&ast.fun.body, &sync, &mapping).unwrap()
    }

    #[test]
    fn empty_sync_points() {
        assert_sync(make_ast(vec![]), BTreeMap::new(), BTreeSet::new());
    }

    #[test]
    fn empty_sync_points_single_par_loop() {
        let x = id("x");
        let ast = make_ast(vec![for_loop(x.clone(), 100, vec![])]);
        let m = GpuMapping::default().add_parallelism(100);
        let mapping = make_mapping(vec![(x.clone(), m)]);
        assert_sync(ast, mapping, BTreeSet::new());
    }

    #[test]
    fn sync_point_in_subsequent_par_loops() {
        let x = id("x");
        let y = id("y");
        let z = id("z");
        let ast = make_ast(vec![for_loop(x.clone(), 24, vec![
            for_loop(y.clone(), 64, vec![]),
            for_loop(z.clone(), 64, vec![])
        ])]);
        let m = GpuMapping {
            grid: LaunchArgs::default()
                .with_blocks_dim(&Dim::X, 24)
                .with_threads_dim(&Dim::X, 64),
            mapping: vec![
                GpuMap::Block {n: 24, dim: Dim::X, mult: 1},
                GpuMap::Thread {n: 64, dim: Dim::X, mult: 1}
            ],
            tpb: DEFAULT_TPB
        };
        let mapping = make_mapping(vec![(x.clone(), m)]);
        let expected = BTreeSet::from([y]);
        assert_sync(ast, mapping, expected);
    }

    #[test]
    #[should_panic]
    fn inter_block_sync_point_err() {
        let x = id("x");
        let y = id("y");
        let z = id("z");
        let ast = make_ast(vec![for_loop(x.clone(), 24, vec![
            for_loop(y.clone(), 2048, vec![]),
            for_loop(z.clone(), 2048, vec![])
        ])]);
        let m = GpuMapping::default().add_parallelism(2048).add_parallelism(24).rev_mapping();
        let mapping = make_mapping(vec![(x.clone(), m)]);
        identify_sync_points(&ast, &mapping).unwrap();
    }
}
