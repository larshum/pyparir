use crate::option::CompileOptions;
use crate::utils::pprint::PrettyPrint;

use std::time;

pub struct DebugEnv {
    debug_perf: bool,
    debug_print: bool,
    start: time::Instant
}

impl DebugEnv {
    pub fn print<T: PrettyPrint>(&self, msg: &str, ast: &T) {
        let bounds = "=".repeat(5);
        if self.debug_perf {
            let now = time::Instant::now();
            let t = now.duration_since(self.start).as_micros();
            println!("{0} {msg} (time: {t} us)", bounds);
        }
        if self.debug_print {
            let ast = ast.pprint_default();
            println!("\n{ast}");
        }
    }
}

pub fn init(opts: &CompileOptions) -> DebugEnv {
    DebugEnv {
        debug_perf: opts.debug_perf,
        debug_print: opts.debug_print,
        start: time::Instant::now()
    }
}
