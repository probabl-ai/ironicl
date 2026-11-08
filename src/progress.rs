//! Progress reports and cancellation for builds and predictions. A caller
//! installs a hook on its thread with [`with_hook`]; builds and predictions
//! report to it between steps, and the hook cancels the work by returning an
//! error, such as [`Error::Interrupted`](crate::error::Error::Interrupted).

use std::{cell::RefCell, rc::Rc};

use crate::error::Result;

/// What a report counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Estimators whose context is built.
    Build,
    /// Query rows predicted.
    Predict,
}

impl Stage {
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Predict => "predict",
        }
    }
}

/// Receives `(stage, done, total)`; an error stops the work and is returned.
pub type Hook = Rc<dyn Fn(Stage, usize, usize) -> Result<()>>;

thread_local! {
    static HOOK: RefCell<Option<Hook>> = const { RefCell::new(None) };
}

/// Run `work` with `hook` receiving this thread's reports, then restore the
/// previous hook, even if `work` panics.
pub fn with_hook<T>(hook: Hook, work: impl FnOnce() -> T) -> T {
    struct Restore(Option<Hook>);
    impl Drop for Restore {
        fn drop(&mut self) {
            HOOK.with(|current| *current.borrow_mut() = self.0.take());
        }
    }
    let _restore = Restore(HOOK.with(|current| current.borrow_mut().replace(hook)));
    work()
}

/// Report `done` of `total` for `stage` to this thread's hook, if any.
///
/// # Errors
///
/// Returns the hook's error, which cancels the work.
pub(crate) fn report(stage: Stage, done: usize, total: usize) -> Result<()> {
    // Clone the hook out so it can run work that reports in turn.
    match HOOK.with(|current| current.borrow().clone()) {
        Some(hook) => hook(stage, done, total),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::error::Error;

    #[test]
    fn reports_reach_the_hook_and_errors_cancel() {
        let seen = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&seen);
        let hook: Hook = Rc::new(move |stage, done, total| {
            sink.borrow_mut().push((stage, done, total));
            if done == 2 {
                Err(Error::Interrupted)
            } else {
                Ok(())
            }
        });
        let result = with_hook(hook, || (0..4).try_for_each(|done| report(Stage::Build, done, 4)));
        assert_eq!(result, Err(Error::Interrupted));
        assert_eq!(
            *seen.borrow(),
            [(Stage::Build, 0, 4), (Stage::Build, 1, 4), (Stage::Build, 2, 4)]
        );
        // The hook is gone once the work returns.
        assert_eq!(report(Stage::Build, 3, 4), Ok(()));
        assert_eq!(seen.borrow().len(), 3);
    }
}
