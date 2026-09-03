//! Opt-in internal phase timing for kernel development.

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Category {
    Column,
    Row,
    Icl,
    Block,
    Affine,
    LayerNorm,
    Gelu,
    Qass,
    Rope,
    Attention,
    AttentionScore,
    AttentionSoftmax,
    AttentionValue,
    AttentionFinalize,
}

#[cfg(feature = "profile")]
impl Category {
    const COUNT: usize = 14;
    const ALL: [Self; Self::COUNT] = [
        Self::Column,
        Self::Row,
        Self::Icl,
        Self::Block,
        Self::Affine,
        Self::LayerNorm,
        Self::Gelu,
        Self::Qass,
        Self::Rope,
        Self::Attention,
        Self::AttentionScore,
        Self::AttentionSoftmax,
        Self::AttentionValue,
        Self::AttentionFinalize,
    ];

    const fn name(self) -> &'static str {
        match self {
            Self::Column => "column",
            Self::Row => "row",
            Self::Icl => "icl",
            Self::Block => "block",
            Self::Affine => "affine",
            Self::LayerNorm => "layer_norm",
            Self::Gelu => "gelu",
            Self::Qass => "qass",
            Self::Rope => "rope",
            Self::Attention => "attention",
            Self::AttentionScore => "attention_score",
            Self::AttentionSoftmax => "attention_softmax",
            Self::AttentionValue => "attention_value",
            Self::AttentionFinalize => "attention_finalize",
        }
    }
}

#[cfg(feature = "profile")]
mod implementation {
    use std::{
        sync::{
            OnceLock,
            atomic::{AtomicU64, Ordering},
        },
        time::Instant,
    };

    use super::Category;

    static ENABLED: OnceLock<bool> = OnceLock::new();
    static NANOS: [AtomicU64; Category::COUNT] = [const { AtomicU64::new(0) }; Category::COUNT];
    static CALLS: [AtomicU64; Category::COUNT] = [const { AtomicU64::new(0) }; Category::COUNT];

    pub struct Scope {
        category: Category,
        start: Option<Instant>,
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            let Some(start) = self.start else { return };
            let index = self.category as usize;
            let nanos = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            NANOS[index].fetch_add(nanos, Ordering::Relaxed);
            CALLS[index].fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    pub fn scope(category: Category) -> Scope {
        let enabled = *ENABLED.get_or_init(|| std::env::var_os("IRONICL_PROFILE").is_some());
        Scope {
            category,
            start: enabled.then(Instant::now),
        }
    }

    pub fn reset() {
        for value in NANOS.iter().chain(&CALLS) {
            value.store(0, Ordering::Relaxed);
        }
    }

    pub fn report() {
        if !*ENABLED.get_or_init(|| std::env::var_os("IRONICL_PROFILE").is_some()) {
            return;
        }
        for category in Category::ALL {
            let index = category as usize;
            let nanos = NANOS[index].load(Ordering::Relaxed);
            let calls = CALLS[index].load(Ordering::Relaxed);
            #[allow(clippy::cast_precision_loss)]
            let milliseconds = nanos as f64 / 1_000_000.0;
            eprintln!(
                "profile_{}_ms: {:.6} calls={calls}",
                category.name(),
                milliseconds
            );
        }
    }
}

#[cfg(not(feature = "profile"))]
mod implementation {
    use super::Category;

    pub struct Scope;

    #[inline]
    #[must_use]
    pub const fn scope(_category: Category) -> Scope {
        Scope
    }

    pub const fn reset() {}
    pub const fn report() {}
}

pub use implementation::{report, reset, scope};
