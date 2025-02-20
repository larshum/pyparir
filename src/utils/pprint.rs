use crate::utils::name::Name;

use itertools::Itertools;
use rand::distributions::{Alphanumeric, DistString};

use std::collections::{BTreeMap, BTreeSet};

pub const DEFAULT_INDENT: usize = 2;

#[derive(Debug)]
pub struct PrettyPrintEnv {
    strs: BTreeSet<String>,
    vars: BTreeMap<Name, String>,
    indent: usize,
    indent_increment: usize,
}

impl PrettyPrintEnv {
    pub fn new() -> Self {
        PrettyPrintEnv {
            strs: BTreeSet::new(),
            vars: BTreeMap::new(),
            indent: 0,
            indent_increment: DEFAULT_INDENT,
        }
    }

    pub fn incr_indent(self) -> Self {
        let indent = self.indent + self.indent_increment;
        PrettyPrintEnv {indent, ..self}
    }

    pub fn decr_indent(self) -> Self {
        let indent = self.indent - self.indent_increment;
        PrettyPrintEnv {indent, ..self}
    }

    pub fn print_indent(&self) -> String {
        " ".repeat(self.indent)
    }
}

pub trait PrettyPrint {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String);

    fn pprint_default(&self) -> String {
        let (_, s) = self.pprint(PrettyPrintEnv::new());
        s
    }
}

fn rand_alphanum(n: usize) -> String {
    Alphanumeric.sample_string(&mut rand::thread_rng(), n)
}

fn alloc_free_string(mut env: PrettyPrintEnv, id: &Name) -> (PrettyPrintEnv, String) {
    let mut s = id.get_str().clone();
    if env.strs.contains(&s) {
        s = id.print_with_sym();
        while env.strs.contains(&s) {
            s = format!("{0}_{1}", id.get_str(), rand_alphanum(5));
        }
    }
    env.strs.insert(s.clone());
    env.vars.insert(id.clone(), s.clone());
    (env, s)
}

impl PrettyPrint for Name {
    fn pprint(&self, env: PrettyPrintEnv) -> (PrettyPrintEnv, String) {
        if let Some(s) = env.vars.get(&self) {
            let s = s.clone();
            (env, s)
        } else {
            alloc_free_string(env, self)
        }
    }
}

pub fn pprint_iter<'a, T: PrettyPrint + 'a, I: Iterator<Item=&'a T>>(
    it: I,
    env: PrettyPrintEnv,
    separator: &str
) -> (PrettyPrintEnv, String) {
    let (env, strs) = it.fold((env, vec![]), |(env, mut strs), v| {
            let (env, v) = v.pprint(env);
            strs.push(v);
            (env, strs)
        });
    (env, strs.into_iter().join(separator))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_distinct_names_print() {
        let n1 = Name::sym_str("x");
        let n2 = n1.clone().with_new_sym();
        let (env, s1) = n1.pprint(PrettyPrintEnv::new());
        let (env, s2) = n2.pprint(env);
        assert_eq!(env.strs.len(), 2);
        assert!(s1 != s2);
        let (_, s3) = n2.pprint(env);
        assert_eq!(s2, s3);
    }
}
