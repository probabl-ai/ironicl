//! How a context's prediction uses device memory: whether its ICL cache stays
//! on the device or streams from the host, and how many query rows each run
//! and each pass take.

use super::{COLUMN_DIM, ICL_BLOCKS, ICL_DIM, RESERVED};

/// Most query rows per run.
const MAX_RUN: usize = 1024;
/// Most query rows per pass over a streamed ICL cache.
const MAX_PASS: usize = 65_536;
/// Fewest query rows per run.
pub(super) const MIN_RUN: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Plan {
    /// The ICL cache streams from the host block by block on every pass.
    pub(super) streamed: bool,
    /// Query rows per run: the column, row, and ICL stages' batch.
    pub(super) run_rows: usize,
    /// Query rows per pass over the ICL blocks: one run when the cache stays
    /// on the device, many when it streams, so each copy serves many rows.
    pub(super) pass_rows: usize,
}

/// Device bytes one query row of a run takes, with `(B, H)` queries: the row
/// stage's four `[B, 4 + H, 128]` buffers and its `[B, 4 + H, 384]`
/// projections dominate, plus the row's `[B, 512]` state between ICL blocks.
pub(super) const fn row_bytes((b, h): (usize, usize)) -> usize {
    4 * (7 * b * (h + RESERVED) * COLUMN_DIM + 17 * b * ICL_DIM)
}

/// The plan for `budget` bytes of device memory with `(B, H)` queries and ICL
/// blocks of `block_bytes` keys and values each, or `None` when not even two
/// streamed blocks fit with room for runs of [`MIN_RUN`] rows.
pub(super) fn plan(budget: usize, (b, h): (usize, usize), block_bytes: usize) -> Option<Plan> {
    let row_qkv = b.checked_mul(h + RESERVED)?.checked_mul(3 * COLUMN_DIM)?;
    let max_run = (u32::MAX as usize / row_qkv).min(MAX_RUN);
    if max_run < MIN_RUN {
        return None;
    }
    let row = row_bytes((b, h));
    let minimum = 2 * MIN_RUN * row;
    let resident = ICL_BLOCKS * block_bytes;
    let streamed = 2 * block_bytes;
    let (streams, cache) = if resident + minimum <= budget {
        (false, resident)
    } else if streamed + minimum <= budget {
        (true, streamed)
    } else {
        return None;
    };
    // Half of what the cache leaves for runs; a streamed pass's state takes up
    // to a quarter more.
    let workspace = budget - cache;
    let run_rows = (workspace / 2 / row).clamp(MIN_RUN, max_run) / MIN_RUN * MIN_RUN;
    let pass_rows = if streams {
        (workspace / 4 / (4 * b * ICL_DIM)).clamp(run_rows, MAX_PASS) / run_rows * run_rows
    } else {
        run_rows
    };
    Some(Plan {
        streamed: streams,
        run_rows,
        pass_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHAPE: (usize, usize) = (8, 20);
    const BLOCK: usize = 100 << 20; // 100 MiB of keys and values per block

    #[test]
    fn a_cache_that_fits_stays_on_the_device() {
        let plan = plan(4 << 30, SHAPE, BLOCK).expect("fits");
        assert!(!plan.streamed);
        assert_eq!(plan.run_rows, MAX_RUN);
        assert_eq!(plan.pass_rows, plan.run_rows);
    }

    #[test]
    fn a_cache_that_does_not_fit_streams_in_whole_runs() {
        let plan = plan(800 << 20, SHAPE, BLOCK).expect("streams");
        assert!(plan.streamed);
        assert!(plan.run_rows >= MIN_RUN && plan.run_rows.is_multiple_of(MIN_RUN));
        assert!(plan.pass_rows >= plan.run_rows && plan.pass_rows.is_multiple_of(plan.run_rows));
    }

    #[test]
    fn without_room_for_two_blocks_nothing_runs() {
        assert_eq!(plan(2 * BLOCK, SHAPE, BLOCK), None);
    }
}
