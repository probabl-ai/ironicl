//! NVIDIA's Kumo-Tabular: its released checkpoints, their architecture
//! values, and the exact tensors each holds.
//!
//! Port of <https://github.com/NVIDIA/structured-data-models/blob/3fdf108a1f1b719b1b983373b71bbba79fcab7e6/sdm/models/kumo/tabular/model.py#L24-L64>.
//! The weights are under OpenMDW-1.1; see `docs/kumo-tabular.md`.

use std::{collections::BTreeMap, fmt};

use crate::TaskKind;

/// A model size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Size {
    Small,
    Medium,
    Large,
}

impl Size {
    pub const ALL: [Self; 3] = [Self::Small, Self::Medium, Self::Large];

    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
        }
    }

    /// The architecture values of this size.
    #[must_use]
    pub fn config(self) -> Config {
        let (cell_channels, embedding_layers, inducing_points, icl_channels, icl_layers, icl_heads, kv_heads) =
            match self {
                Self::Small => (128, 4, 128, 512, 12, 8, None),
                Self::Medium => (256, 6, 256, 512, 24, 8, Some(2)),
                Self::Large => (256, 6, 256, 1024, 24, 16, Some(2)),
            };
        Config {
            cell_channels,
            embedding_layers,
            embedding_heads: 4,
            inducing_points,
            group_size: 3,
            frequencies: 32,
            readout_tokens: 4,
            icl_channels,
            icl_layers,
            icl_heads,
            query_key_value_heads: kv_heads,
        }
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A released checkpoint: a size and a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Release {
    pub size: Size,
    pub task: TaskKind,
}

impl Release {
    /// Every released checkpoint.
    pub const ALL: [Self; 6] = [
        Self::new(Size::Small, TaskKind::Classifier),
        Self::new(Size::Small, TaskKind::Regressor),
        Self::new(Size::Medium, TaskKind::Classifier),
        Self::new(Size::Medium, TaskKind::Regressor),
        Self::new(Size::Large, TaskKind::Classifier),
        Self::new(Size::Large, TaskKind::Regressor),
    ];

    #[must_use]
    pub const fn new(size: Size, task: TaskKind) -> Self {
        Self { size, task }
    }

    /// The model's outputs: 10 class logits or 999 quantiles.
    #[must_use]
    pub fn outputs(self) -> u64 {
        match self.task {
            TaskKind::Classifier => 10,
            TaskKind::Regressor => 999,
        }
    }

    /// Every tensor the checkpoint holds, by name, with its dimensions.
    #[must_use]
    pub fn manifest(self) -> BTreeMap<String, Vec<u64>> {
        let config = self.size.config();
        let (cell, icl) = (config.cell_channels, config.icl_channels);
        let head_dim = cell / config.embedding_heads;
        let mut tensors = BTreeMap::new();
        let mut add = |name: String, dimensions: &[u64]| {
            tensors.insert(name, dimensions.to_vec());
        };
        let embedding = "row_embedding.cell_embedding";
        let frequencies = [config.group_size, config.frequencies];
        add(format!("{embedding}.num_freq"), &frequencies);
        add(format!("{embedding}.cat_freq"), &frequencies);
        for linear in ["num_lin", "cat_lin"] {
            add(
                format!("{embedding}.{linear}.weight"),
                &[cell, 2 * config.frequencies],
            );
            add(format!("{embedding}.{linear}.bias"), &[cell]);
        }
        add(format!("{embedding}.nan_lin.weight"), &[cell, config.group_size]);
        match self.task {
            TaskKind::Classifier => {
                add("row_embedding.y_emb.weight".into(), &[10, cell]);
                add("icl_block.y_emb.weight".into(), &[10, icl]);
            }
            TaskKind::Regressor => {
                add("row_embedding.y_lin.weight".into(), &[cell, 1]);
                add("icl_block.y_lin.weight".into(), &[icl, 1]);
            }
        }
        add(
            "row_embedding.readout_token".into(),
            &[config.readout_tokens, cell],
        );
        for layer in 0..config.embedding_layers {
            let column = format!("row_embedding.col_blocks.{layer}");
            add(
                format!("{column}.inducing_points"),
                &[config.inducing_points, cell],
            );
            block(&mut add, &format!("{column}.inducing_block"), cell);
            add(
                format!("{column}.inducing_block.attn.sdpa.query_scaling.head_scale"),
                &[config.embedding_heads],
            );
            block(&mut add, &format!("{column}.output_block"), cell);
            let row = format!("row_embedding.row_blocks.{layer}");
            block(&mut add, &row, cell);
            for transform in ["query_transform", "key_transform"] {
                add(format!("{row}.attn.{transform}.0.inv_freq"), &[head_dim / 2]);
            }
            let scaling = format!("{row}.attn.sdpa.query_scaling");
            add(format!("{scaling}.head_scale"), &[config.embedding_heads]);
            add(format!("{scaling}.gate.0.weight"), &[64, head_dim]);
            add(format!("{scaling}.gate.0.bias"), &[64]);
            add(format!("{scaling}.gate.2.weight"), &[head_dim, 64]);
            add(format!("{scaling}.gate.2.bias"), &[head_dim]);
        }
        add("row_embedding.norm.weight".into(), &[cell]);
        let row_width = config.readout_tokens * cell;
        if row_width != icl {
            add("row_project.weight".into(), &[icl, row_width]);
            add("row_project.bias".into(), &[icl]);
        }
        for layer in 0..config.icl_layers {
            let name = format!("icl_block.layers.{layer}");
            block(&mut add, &name, icl);
            add(
                format!("{name}.attn.sdpa.query_scaling.head_scale"),
                &[config.icl_heads],
            );
        }
        add("icl_block.norm.weight".into(), &[icl]);
        add("icl_block.head.0.weight".into(), &[2 * icl, icl]);
        add("icl_block.head.0.bias".into(), &[2 * icl]);
        add("icl_block.head.2.weight".into(), &[self.outputs(), 2 * icl]);
        add("icl_block.head.2.bias".into(), &[self.outputs()]);
        tensors
    }
}

impl fmt::Display for Release {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Kumo-Tabular {} {}", self.size, self.task)
    }
}

impl std::str::FromStr for Release {
    type Err = String;

    /// A release by its `<size>-<task>` name, such as `large-classifier`.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|release| format!("{}-{}", release.size, release.task) == name)
            .ok_or_else(|| format!("{name} is not <small|medium|large>-<classifier|regressor>"))
    }
}

/// A transformer block's tensors: RMS norms on the query and on the keys and
/// values, a fused QKV projection, the output projection, and an MLP that
/// widens twofold after its own RMS norm.
fn block(add: &mut impl FnMut(String, &[u64]), name: &str, width: u64) {
    add(format!("{name}.query_norm.weight"), &[width]);
    add(format!("{name}.key_value_norm.weight"), &[width]);
    add(format!("{name}.attn.qkv_lin.weight"), &[3 * width, width]);
    add(format!("{name}.attn.qkv_lin.bias"), &[3 * width]);
    add(format!("{name}.attn.out_lin.weight"), &[width, width]);
    add(format!("{name}.attn.out_lin.bias"), &[width]);
    add(format!("{name}.mlp.0.weight"), &[width]);
    add(format!("{name}.mlp.1.weight"), &[2 * width, width]);
    add(format!("{name}.mlp.1.bias"), &[2 * width]);
    add(format!("{name}.mlp.3.weight"), &[width, 2 * width]);
    add(format!("{name}.mlp.3.bias"), &[width]);
}

/// A size's architecture values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Each cell's embedding width in the column and row stages.
    pub cell_channels: u64,
    /// Interleaved column and row layers.
    pub embedding_layers: u64,
    pub embedding_heads: u64,
    /// The column stage's inducing points per layer.
    pub inducing_points: u64,
    /// Columns embedded together, at offsets 0, 1, and 3.
    pub group_size: u64,
    /// Fourier frequencies per cell.
    pub frequencies: u64,
    /// Readout tokens per row.
    pub readout_tokens: u64,
    pub icl_channels: u64,
    pub icl_layers: u64,
    pub icl_heads: u64,
    /// The key and value heads test rows attend with, or `None` for all.
    pub query_key_value_heads: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::{Release, Size};
    use crate::TaskKind;

    #[test]
    fn manifests_match_the_released_checkpoints() {
        // Tensor and element counts of each checkpoint file, inv_freq buffers included.
        for (size, task, tensors, elements) in [
            (Size::Small, TaskKind::Classifier, 328, 27_458_378),
            (Size::Small, TaskKind::Regressor, 328, 28_466_343),
            (Size::Medium, TaskKind::Classifier, 558, 61_485_626),
            (Size::Medium, TaskKind::Regressor, 558, 62_492_439),
            (Size::Large, TaskKind::Classifier, 556, 213_668_602),
            (Size::Large, TaskKind::Regressor, 556, 215_683_543),
        ] {
            let manifest = Release::new(size, task).manifest();
            let total: u64 = manifest
                .values()
                .map(|dimensions| dimensions.iter().product::<u64>())
                .sum();
            assert_eq!((manifest.len(), total), (tensors, elements), "{size} {task}");
        }
    }
}
