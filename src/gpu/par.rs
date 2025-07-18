/// Functions defining how to map the parallel specification of for-loops in a Python function to a
/// GPU grid using a straightforward approach.

use super::ast::{Dim, LaunchArgs};
use crate::par;
use crate::prickle_compile_error;
use crate::ir::ast::*;
use crate::utils::err::*;
use crate::utils::info::Info;
use crate::utils::name::Name;
use crate::utils::smap::SFold;

use itertools::Itertools;

use std::collections::BTreeMap;

pub const WARP_SIZE: i64 = 32;

#[derive(Debug, PartialEq)]
pub struct ParEntry {
    tpb: i64,
    p: Vec<i64>
}

impl ParEntry {
    pub fn new(tpb: i64, p: Vec<i64>) -> Self {
        ParEntry {tpb, p}
    }
}

type ParResult = CompileResult<BTreeMap<Name, ParEntry>>;

struct Par {
    n: Vec<i64>,
    tpb: i64,
    i: Info
}

impl Par {
    pub fn new(n: Vec<i64>, i: Info) -> Par {
        Par {n, tpb: par::DEFAULT_TPB, i}
    }
}

impl PartialEq for Par {
    fn eq(&self, other: &Self) -> bool {
        self.n.eq(&other.n) && (self.tpb.eq(&other.tpb))
    }
}

// Attempts to unify the parallel structure of two statements. An empty structure represents
// sequential code. We can unify it with any kind of parallelization. On the other hand, two
// parallel structures have to be equivalent to merge them.
fn unify_par(acc: Par, p: Par) -> CompileResult<Par> {
    if acc.n.is_empty() {
        Ok(p)
    } else if p.n.is_empty() {
        Ok(acc)
    } else if acc.n.eq(&p.n) {
        Ok(acc)
    } else {
        let msg = concat!(
            "The parallel structure of this statement is inconsistent relative to ",
            "previous statements on the same level of nesting"
        );
        prickle_compile_error!(p.i, "{}", msg)
    }
}

fn find_parallel_structure_stmt_par(stmt: &Stmt) -> CompileResult<Par> {
    match stmt {
        Stmt::Definition {i, ..} | Stmt::Assign {i, ..} | Stmt::Return {i, ..} |
        Stmt::SyncPoint {i, ..} | Stmt::Alloc {i, ..} | Stmt::Free {i, ..} => {
            Ok(Par::new(vec![], i.clone()))
        },
        Stmt::If {thn, els, i, ..} => {
            let thn = find_parallel_structure_stmts_par(thn)?;
            let els = find_parallel_structure_stmts_par(els)?;
            // For if-conditions, we require that both branches have exactly the same parallel
            // structure. That is, we do not allow parallelism only in one branch - this is due to
            // the performance implications this might have.
            if thn.n.eq(&els.n) {
                Ok(Par::new(thn.n, i.clone()))
            } else {
                let lhs = thn.n.iter().join(", ");
                let rhs = els.n.iter().join(", ");
                let msg = format!(
                    "Found branches with inconsistent parallel structure: \
                     [{lhs}] != [{rhs}].\n\
                     This is not supported because the compiller cannot generate \
                     efficient code for this."
                );
                prickle_compile_error!(i, "{}", msg)
            }
        },
        Stmt::For {body, par, ..} => {
            let mut inner_par = find_parallel_structure_stmts_par(body)?;
            if par.is_parallel() {
                inner_par.n.insert(0, par.nthreads);
            };
            Ok(inner_par)
        },
        Stmt::While {body, ..} => {
            find_parallel_structure_stmts_par(body)
        },
    }
}

fn find_parallel_structure_stmts_par(stmts: &Vec<Stmt>) -> CompileResult<Par> {
    let p = Par::new(vec![], Info::default());
    stmts.iter()
        .map(find_parallel_structure_stmt_par)
        .fold(Ok(p), |acc, stmt_par| unify_par(acc?, stmt_par?))
}

fn find_parallel_structure_stmt_seq(
    mut acc: BTreeMap<Name, ParEntry>,
    stmt: &Stmt
) -> ParResult {
    match stmt {
        Stmt::For {var, body, par, ..} if par.is_parallel() => {
            let mut p = find_parallel_structure_stmts_par(body)?;
            p.n.insert(0, par.nthreads);
            // Ensure that the innermost thread count of the parallel structure of this loop uses a
            // thread count evenly divisible by the size of a warp. This is very important because
            // warp-level intrinsics behave unexpectedly when not all threads of a warp are used,
            // causing parallel reductions to misbehave.
            match p.n.last_mut() {
                Some(n) => *n = ((*n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE,
                None => ()
            };
            acc.insert(var.clone(), ParEntry::new(par.tpb, p.n));
            Ok(acc)
        },
        Stmt::For {body, ..} => find_parallel_structure_stmts_seq(acc, body),
        Stmt::Definition {..} | Stmt::Assign {..} | Stmt::SyncPoint {..} |
        Stmt::While {..} | Stmt::If {..} | Stmt::Return {..} | Stmt::Alloc {..} |
        Stmt::Free {..} => {
            stmt.sfold_result(Ok(acc), find_parallel_structure_stmt_seq)
        }
    }
}

fn find_parallel_structure_stmts_seq(
    acc: BTreeMap<Name, ParEntry>,
    stmts: &Vec<Stmt>
) -> ParResult {
    stmts.sfold_result(Ok(acc), find_parallel_structure_stmt_seq)
}

fn find_parallel_structure_fun_def(fun_def: &FunDef) -> ParResult {
    find_parallel_structure_stmts_seq(BTreeMap::new(), &fun_def.body)
}

pub fn find_parallel_structure(ast: &Ast) -> ParResult {
    // NOTE: The compiler assumes only the main function definition (the last one) involves any
    // parallelism.
    let def = ast.defs.last().unwrap();
    find_parallel_structure_fun_def(&def)
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuMap {
    Block {n: i64, mult: i64, dim: Dim},
    Thread {n: i64, mult: i64, dim: Dim},
    ThreadBlock {n: i64, nthreads: i64, nblocks: i64, dim: Dim},
}

#[derive(Clone, Debug, PartialEq)]
pub struct GpuMapping {
    pub grid: LaunchArgs,
    pub mapping: Vec<GpuMap>,
    pub tpb: i64
}

impl GpuMapping {
    pub fn new(tpb: i64) -> Self {
        GpuMapping {
            grid: LaunchArgs::default(),
            mapping: vec![],
            tpb
        }
    }

    pub fn add_parallelism(self, n: i64) -> Self {
        match self.mapping.last() {
            Some(_) => self.add_block_par(n),
            None => {
                if n > self.tpb {
                    let GpuMapping {grid, mut mapping, tpb} = self;
                    let nblocks = (n + tpb - 1) / tpb;
                    let grid = grid
                        .with_blocks_dim(&Dim::X, nblocks)
                        .with_threads_dim(&Dim::X, tpb);
                    mapping.push(GpuMap::ThreadBlock {n, nthreads: tpb, nblocks, dim: Dim::X});
                    GpuMapping {grid, mapping, tpb}
                } else {
                    self.with_thread_par(Dim::X, n)
                }
            },
        }
    }

    pub fn with_thread_par(self, dim: Dim, n: i64) -> Self {
        let GpuMapping {grid, mut mapping, tpb} = self;
        let old_dim_threads = grid.threads.get_dim(&dim);
        let new_dim_threads = old_dim_threads * n;
        let grid = grid.with_threads_dim(&dim, new_dim_threads);
        mapping.push(GpuMap::Thread {n, dim, mult: old_dim_threads});
        GpuMapping {grid, mapping, tpb}
    }

    pub fn add_block_par(self, n: i64) -> Self {
        if n > 65535 {
            self.with_block_par(Dim::X, n)
        } else {
            if self.grid.blocks.y == 1 {
                self.with_block_par(Dim::Y, n)
            } else if self.grid.blocks.z == 1 {
                self.with_block_par(Dim::Z, n)
            } else {
                self.with_block_par(Dim::X, n)
            }
        }
    }

    pub fn with_block_par(self, dim: Dim, n: i64) -> Self {
        let GpuMapping {grid, mut mapping, tpb} = self;
        let old_dim_blocks = grid.blocks.get_dim(&dim);
        let new_dim_blocks = old_dim_blocks * n;
        let grid = grid.with_blocks_dim(&dim, new_dim_blocks);
        mapping.push(GpuMap::Block {n, dim, mult: old_dim_blocks});
        GpuMapping {grid, mapping, tpb}
    }

    pub fn rev_mapping(mut self) -> Self {
        self.mapping.reverse();
        self
    }

    pub fn get_mapping(&self) -> Vec<GpuMap> {
        self.mapping.clone()
    }
}

fn add_par_to_grid(m: GpuMapping, n: i64) -> GpuMapping {
    m.add_parallelism(n)
}

fn par_to_gpu_mapping(par: ParEntry) -> GpuMapping {
    let ParEntry {tpb, p} = par;
    p.into_iter()
        .rev()
        .fold(GpuMapping::new(tpb), add_par_to_grid)
        .rev_mapping()
}

pub fn map_gpu_grid(
    par: BTreeMap<Name, ParEntry>
) -> BTreeMap<Name, GpuMapping> {
    par.into_iter()
        .map(|(id, par)| (id, par_to_gpu_mapping(par)))
        .collect::<BTreeMap<Name, GpuMapping>>()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ir::ir_builder;
    use crate::ir::ir_builder::*;

    fn id(s: &str) -> Name {
        ir_builder::id(s).with_new_sym()
    }

    fn find_structure(def: FunDef) -> BTreeMap<Name, ParEntry> {
        find_parallel_structure_fun_def(&def).unwrap()
    }

    fn to_map<T1: Ord, T2>(v: Vec<(T1, T2)>) -> BTreeMap<T1, T2> {
        v.into_iter().collect::<BTreeMap<T1, T2>>()
    }

    #[test]
    fn seq_loops() {
        let def = fun_def(vec![for_loop(id("x"), 0, vec![for_loop(id("y"), 0, vec![])])]);
        assert_eq!(find_structure(def), BTreeMap::new());
    }

    #[test]
    fn single_par_loop() {
        let x = id("x");
        let def = fun_def(vec![for_loop(x.clone(), 10, vec![])]);
        let expected = to_map(vec![(x, ParEntry::new(par::DEFAULT_TPB, vec![32]))]);
        assert_eq!(find_structure(def), expected);
    }

    #[test]
    fn consistent_par_if() {
        let x = id("x");
        let def = fun_def(vec![for_loop(x.clone(), 2, vec![if_cond(
            vec![for_loop(id("y"), 3, vec![])],
            vec![for_loop(id("z"), 3, vec![for_loop(id("w"), 0, vec![])])],
        )])]);
        let expected = to_map(vec![
            (x, ParEntry::new(par::DEFAULT_TPB, vec![2, 32]))
        ]);
        assert_eq!(find_structure(def), expected);
    }

    #[test]
    fn inconsistent_par_if() {
        let def = fun_def(vec![for_loop(id("x"), 2, vec![if_cond(
            vec![for_loop(id("y"), 3, vec![])],
            vec![for_loop(id("z"), 4, vec![])]
        )])]);
        assert!(find_parallel_structure_fun_def(&def).is_err());
    }

    #[test]
    fn equal_par_but_distinct_structure() {
        let def = fun_def(vec![for_loop(id("x"), 5, vec![
            if_cond(
                vec![for_loop(id("y"), 4, vec![for_loop(id("z"), 7, vec![])])],
                vec![for_loop(id("z"), 7, vec![for_loop(id("y"), 4, vec![])])]
            )
        ])]);
        assert!(find_parallel_structure_fun_def(&def).is_err());
    }

    #[test]
    fn nested_parallelism() {
        let x = id("x");
        let def = fun_def(vec![
            for_loop(x.clone(), 10, vec![for_loop(id("y"), 15, vec![for_loop(id("z"), 0, vec![])])])
        ]);
        let expected = to_map(vec![
            (x, ParEntry::new(par::DEFAULT_TPB, vec![10, 32]))
        ]);
        assert_eq!(find_structure(def), expected);
    }

    #[test]
    fn nested_consistent_if() {
        let x = id("x");
        let def = fun_def(vec![
            for_loop(x.clone(), 10, vec![if_cond(
                vec![for_loop(id("y"), 15, vec![])],
                vec![for_loop(id("y"), 0, vec![for_loop(id("z"), 15, vec![])])]
            )])
        ]);
        let expected = to_map(vec![
            (x, ParEntry::new(par::DEFAULT_TPB, vec![10, 32]))
        ]);
        assert_eq!(find_structure(def), expected);
    }

    #[test]
    fn nested_inconsistent_if() {
        let def = fun_def(vec![
            for_loop(id("x"), 10, vec![if_cond(
                vec![for_loop(id("y"), 10, vec![])],
                vec![for_loop(id("z"), 15, vec![])]
            )])
        ]);
        assert!(find_parallel_structure_fun_def(&def).is_err());
    }

    #[test]
    fn subsequent_par() {
        let x1 = id("x");
        let x2 = id("x");
        let def = fun_def(vec![
            for_loop(x1.clone(), 5, vec![]),
            for_loop(x2.clone(), 7, vec![])
        ]);
        let expected = to_map(vec![
            (x1, ParEntry::new(par::DEFAULT_TPB, vec![32])),
            (x2, ParEntry::new(par::DEFAULT_TPB, vec![32]))
        ]);
        assert_eq!(find_structure(def), expected);
    }

    #[test]
    fn subsequent_distinct_struct_par() {
        let x1 = id("x");
        let x2 = id("x");
        let def = fun_def(vec![
            for_loop(x1.clone(), 5, vec![
                if_cond(
                    vec![for_loop(id("y"), 6, vec![])],
                    vec![for_loop(id("y"), 0, vec![for_loop(id("z"), 6, vec![])])]
                )
            ]),
            for_loop(x2.clone(), 7, vec![for_loop(id("w"), 42, vec![])])
        ]);
        let expected = to_map(vec![
            (x1, ParEntry::new(par::DEFAULT_TPB, vec![5, 32])),
            (x2, ParEntry::new(par::DEFAULT_TPB, vec![7, 64]))
        ]);
        assert_eq!(find_structure(def), expected);
    }

    fn assert_par_mapping(par: ParEntry, mapping: GpuMapping) {
        let x = id("x");
        let par = to_map(vec![(x.clone(), par)]);
        let expected = to_map(vec![(x, mapping)]);
        assert_eq!(map_gpu_grid(par), expected);
    }

    #[test]
    fn map_empty_grid() {
        assert_eq!(map_gpu_grid(BTreeMap::new()), BTreeMap::new());
    }

    #[test]
    fn map_threads_single_par() {
        let par = ParEntry::new(par::DEFAULT_TPB, vec![128]);
        let mapping = GpuMapping {
            grid: LaunchArgs::default().with_threads_dim(&Dim::X, 128),
            mapping: vec![GpuMap::Thread {n: 128, dim: Dim::X, mult: 1}],
            tpb: par::DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_many_threads_to_grid() {
        let par = ParEntry::new(par::DEFAULT_TPB, vec![2048]);
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(&Dim::X, 1024)
                .with_blocks_dim(&Dim::X, 2),
            mapping: vec![GpuMap::ThreadBlock {n: 2048, nthreads: 1024, nblocks: 2, dim: Dim::X}],
            tpb: par::DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_many_threads_to_grid_custom_tpb() {
        let par = ParEntry::new(512, vec![2048]);
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(&Dim::X, 512)
                .with_blocks_dim(&Dim::X, 4),
            mapping: vec![GpuMap::ThreadBlock {n: 2048, nthreads: 512, nblocks: 4, dim: Dim::X}],
            tpb: 512
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_par_threads_and_blocks() {
        let par = ParEntry::new(par::DEFAULT_TPB, vec![64, 128]);
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(&Dim::X, 128)
                .with_blocks_dim(&Dim::Y, 64),
            mapping: vec![
                GpuMap::Block {n: 64, dim: Dim::Y, mult: 1},
                GpuMap::Thread {n: 128, dim: Dim::X, mult: 1}
            ],
            tpb: par::DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_multi_blocks() {
        let par = ParEntry::new(par::DEFAULT_TPB, vec![512, 128, 1682]);
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(&Dim::X, 1024)
                .with_blocks_dim(&Dim::X, 2)
                .with_blocks_dim(&Dim::Y, 128)
                .with_blocks_dim(&Dim::Z, 512),
            mapping: vec![
                GpuMap::Block {n: 512, dim: Dim::Z, mult: 1},
                GpuMap::Block {n: 128, dim: Dim::Y, mult: 1},
                GpuMap::ThreadBlock {n: 1682, nthreads: 1024, nblocks: 2, dim: Dim::X}
            ],
            tpb: par::DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }

    #[test]
    fn map_overlapping_dim_blocks() {
        let par = ParEntry::new(par::DEFAULT_TPB, vec![64, 512, 128, 1682]);
        let mapping = GpuMapping {
            grid: LaunchArgs::default()
                .with_threads_dim(&Dim::X, 1024)
                .with_blocks_dim(&Dim::X, 2*64)
                .with_blocks_dim(&Dim::Y, 128)
                .with_blocks_dim(&Dim::Z, 512),
            mapping: vec![
                GpuMap::Block {n: 64, dim: Dim::X, mult: 2},
                GpuMap::Block {n: 512, dim: Dim::Z, mult: 1},
                GpuMap::Block {n: 128, dim: Dim::Y, mult: 1},
                GpuMap::ThreadBlock {n: 1682, nthreads: 1024, nblocks: 2, dim: Dim::X}
            ],
            tpb: par::DEFAULT_TPB
        };
        assert_par_mapping(par, mapping);
    }
}
