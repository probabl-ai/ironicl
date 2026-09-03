#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::manual_is_multiple_of,
    clippy::missing_errors_doc,
    clippy::needless_range_loop,
    clippy::too_many_arguments
)]

use crate::error::{Error, Result};
use crate::profile::{self, Category};

const LAYER_NORM_EPSILON: f32 = 1e-5;
const SQRT_2: f32 = core::f32::consts::SQRT_2;

/// Borrowed FP32 parameters for elementwise QASSMax.
///
/// `base_*` describe `Linear(1, hidden) -> GELU -> Linear(hidden, embed_dim)`.
/// `query_*` describe `Linear(head_dim, hidden) -> GELU -> Linear(hidden, head_dim)`.
#[derive(Debug, Clone, Copy)]
pub struct QassMaxWeights<'a> {
    pub base_first_weight: &'a [f32],
    pub base_first_bias: &'a [f32],
    pub base_second_weight: &'a [f32],
    pub base_second_bias: &'a [f32],
    pub query_first_weight: &'a [f32],
    pub query_first_bias: &'a [f32],
    pub query_second_weight: &'a [f32],
    pub query_second_bias: &'a [f32],
}

fn checked_elements(rows: usize, columns: usize, name: &'static str) -> Result<usize> {
    if rows == 0 || columns == 0 {
        return Err(Error::InvalidArgument(format!(
            "{name} dimensions must be non-zero"
        )));
    }
    rows.checked_mul(columns).ok_or(Error::ArithmeticOverflow(name))
}

fn require_length(values: &[f32], expected: usize, name: &str) -> Result<()> {
    if values.len() != expected {
        return Err(Error::InvalidArgument(format!(
            "{name} has length {}; expected {expected}",
            values.len()
        )));
    }
    Ok(())
}

fn gelu(value: f32) -> f32 {
    0.5 * value * (1.0 + libm::erff(value / SQRT_2))
}

fn qass_tanh(value: f32) -> f32 {
    value.tanh()
}

/// Deterministic scalar dot product used as an initial correctness oracle.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when the input lengths differ.
pub fn dot_f32(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err(Error::InvalidArgument(format!(
            "dot-product lengths differ: {} and {}",
            left.len(),
            right.len()
        )));
    }
    Ok(left
        .iter()
        .zip(right)
        .fold(0.0_f32, |sum, (left, right)| left.mul_add(*right, sum)))
}

/// Apply a PyTorch `Linear` layer to `rows` row-major input rows.
///
/// `weight` has PyTorch layout `[output_dim, input_dim]`. `bias`, when present,
/// has `output_dim` elements. `output` is overwritten and has `[rows, output_dim]`.
pub fn affine_f32(
    input: &[f32],
    rows: usize,
    input_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    let _profile = profile::scope(Category::Affine);
    let input_len = checked_elements(rows, input_dim, "affine input")?;
    let weight_len = checked_elements(output_dim, input_dim, "affine weight")?;
    let output_len = checked_elements(rows, output_dim, "affine output")?;
    require_length(input, input_len, "affine input")?;
    require_length(weight, weight_len, "affine weight")?;
    if let Some(bias) = bias {
        require_length(bias, output_dim, "affine bias")?;
    }
    require_length(output, output_len, "affine output")?;

    if super::native::affine_f32(input, rows, input_dim, weight, bias, output_dim, output) {
        return Ok(());
    }

    for (input_row, output_row) in input
        .chunks_exact(input_dim)
        .zip(output.chunks_exact_mut(output_dim))
    {
        for (out_index, value) in output_row.iter_mut().enumerate() {
            let initial = bias.map_or(0.0, |values| values[out_index]);
            *value = weight[out_index * input_dim..][..input_dim]
                .iter()
                .zip(input_row)
                .fold(initial, |sum, (weight, input)| weight.mul_add(*input, sum));
        }
    }
    Ok(())
}

/// Apply a `3 -> output_dim` affine layer with input-major `[3, output_dim]`
/// weights. This layout lets the native kernel vectorize across output channels.
pub(crate) fn affine_input3_transposed_f32(
    input: &[f32],
    rows: usize,
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    let _profile = profile::scope(Category::Affine);
    require_length(input, checked_elements(rows, 3, "affine input")?, "affine input")?;
    require_length(
        weight,
        checked_elements(3, output_dim, "affine weight")?,
        "affine weight",
    )?;
    require_length(bias, output_dim, "affine bias")?;
    require_length(
        output,
        checked_elements(rows, output_dim, "affine output")?,
        "affine output",
    )?;
    if super::native::affine_input3_transposed_f32(input, rows, weight, bias, output_dim, output) {
        return Ok(());
    }
    for (input, output) in input.chunks_exact(3).zip(output.chunks_exact_mut(output_dim)) {
        for channel in 0..output_dim {
            let value = weight[channel].mul_add(input[0], bias[channel]);
            let value = weight[output_dim + channel].mul_add(input[1], value);
            output[channel] = weight[2 * output_dim + channel].mul_add(input[2], value);
        }
    }
    Ok(())
}

/// Look up rows from a row-major embedding table, equivalent to one-hot linear projection.
pub fn embedding_lookup_f32(
    indices: &[usize],
    embeddings: &[f32],
    vocabulary: usize,
    embedding_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    let table_len = checked_elements(vocabulary, embedding_dim, "embedding table")?;
    let output_len = checked_elements(indices.len(), embedding_dim, "embedding output")?;
    require_length(embeddings, table_len, "embedding table")?;
    require_length(output, output_len, "embedding output")?;
    for (&index, output_row) in indices.iter().zip(output.chunks_exact_mut(embedding_dim)) {
        if index >= vocabulary {
            return Err(Error::InvalidArgument(format!(
                "embedding index {index} is outside vocabulary of {vocabulary}"
            )));
        }
        output_row.copy_from_slice(&embeddings[index * embedding_dim..][..embedding_dim]);
    }
    Ok(())
}

/// Apply a one-hot input followed by a PyTorch `Linear` layer.
///
/// `weight` has `[output_dim, classes]` layout. This differs from an embedding
/// lookup: each class selects a column and the linear bias is added.
pub fn one_hot_linear_f32(
    classes: &[usize],
    weight: &[f32],
    bias: &[f32],
    class_count: usize,
    output_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    require_length(
        weight,
        checked_elements(output_dim, class_count, "one-hot linear weight")?,
        "one-hot linear weight",
    )?;
    require_length(bias, output_dim, "one-hot linear bias")?;
    require_length(
        output,
        checked_elements(classes.len(), output_dim, "one-hot linear output")?,
        "one-hot linear output",
    )?;
    for (&class, row) in classes.iter().zip(output.chunks_exact_mut(output_dim)) {
        if class >= class_count {
            return Err(Error::InvalidArgument(format!(
                "one-hot class {class} is outside class count {class_count}"
            )));
        }
        for output_index in 0..output_dim {
            row[output_index] = weight[output_index * class_count + class] + bias[output_index];
        }
    }
    Ok(())
}

/// Apply LayerNorm over the final dimension with PyTorch's default epsilon of `1e-5`.
pub fn layer_norm_f32(
    input: &[f32],
    rows: usize,
    width: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
) -> Result<()> {
    let _profile = profile::scope(Category::LayerNorm);
    let elements = checked_elements(rows, width, "layer norm")?;
    require_length(input, elements, "layer norm input")?;
    require_length(weight, width, "layer norm weight")?;
    if let Some(bias) = bias {
        require_length(bias, width, "layer norm bias")?;
    }
    require_length(output, elements, "layer norm output")?;
    if !exact_layer_norm_enabled() && super::native::layer_norm_f32(input, rows, width, weight, bias, output)
    {
        return Ok(());
    }
    for (input_row, output_row) in input.chunks_exact(width).zip(output.chunks_exact_mut(width)) {
        let mean = input_row.iter().sum::<f32>() / width as f32;
        let variance = input_row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / width as f32;
        let scale = (variance + LAYER_NORM_EPSILON).sqrt().recip();
        for (column, value) in output_row.iter_mut().enumerate() {
            *value = (input_row[column] - mean) * scale * weight[column]
                + bias.map_or(0.0, |values| values[column]);
        }
    }
    Ok(())
}

fn exact_layer_norm_enabled() -> bool {
    static EXACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EXACT.get_or_init(|| std::env::var_os("IRONICL_EXACT_LAYER_NORM").is_some())
}

/// Apply GELU, using a bounded native approximation unless exact mode is requested.
pub fn gelu_f32(input: &[f32], output: &mut [f32]) -> Result<()> {
    let _profile = profile::scope(Category::Gelu);
    require_length(output, input.len(), "GELU output")?;
    if !exact_gelu_enabled() && super::native::gelu_f32(input, output) {
        return Ok(());
    }
    for (input, output) in input.iter().zip(output) {
        *output = gelu(*input);
    }
    Ok(())
}

fn exact_gelu_enabled() -> bool {
    static EXACT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EXACT.get_or_init(|| std::env::var_os("IRONICL_EXACT_GELU").is_some())
}

/// Add equally sized FP32 tensors into a caller-owned output buffer.
pub fn residual_add_f32(left: &[f32], right: &[f32], output: &mut [f32]) -> Result<()> {
    require_length(right, left.len(), "residual right input")?;
    require_length(output, left.len(), "residual output")?;
    for ((left, right), output) in left.iter().zip(right).zip(output) {
        *output = left + right;
    }
    Ok(())
}

/// Project input rows with packed PyTorch Q/K/V parameters and split the result.
pub fn packed_qkv_projection_f32(
    input: &[f32],
    rows: usize,
    embed_dim: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    packed_scratch: &mut [f32],
    query: &mut [f32],
    key: &mut [f32],
    value: &mut [f32],
) -> Result<()> {
    let rows_embed = checked_elements(rows, embed_dim, "QKV output")?;
    let packed_dim = embed_dim
        .checked_mul(3)
        .ok_or(Error::ArithmeticOverflow("QKV dimension"))?;
    require_length(
        weight,
        checked_elements(packed_dim, embed_dim, "QKV weight")?,
        "QKV weight",
    )?;
    if let Some(bias) = bias {
        require_length(bias, packed_dim, "QKV bias")?;
    }
    require_length(query, rows_embed, "query output")?;
    require_length(key, rows_embed, "key output")?;
    require_length(value, rows_embed, "value output")?;
    require_length(
        packed_scratch,
        checked_elements(rows, packed_dim, "packed QKV output")?,
        "packed QKV scratch",
    )?;
    affine_f32(input, rows, embed_dim, weight, bias, packed_dim, packed_scratch)?;
    for (packed_row, ((query_row, key_row), value_row)) in packed_scratch.chunks_exact(packed_dim).zip(
        query
            .chunks_exact_mut(embed_dim)
            .zip(key.chunks_exact_mut(embed_dim))
            .zip(value.chunks_exact_mut(embed_dim)),
    ) {
        query_row.copy_from_slice(&packed_row[..embed_dim]);
        key_row.copy_from_slice(&packed_row[embed_dim..2 * embed_dim]);
        value_row.copy_from_slice(&packed_row[2 * embed_dim..]);
    }
    Ok(())
}

/// Apply elementwise QASSMax to `[heads, rows, head_dim]` queries.
pub fn qassmax_f32(
    queries: &[f32],
    heads: usize,
    rows: usize,
    head_dim: usize,
    source_len: usize,
    hidden_dim: usize,
    weights: QassMaxWeights<'_>,
    base_hidden_scratch: &mut [f32],
    base_scale_scratch: &mut [f32],
    query_hidden_scratch: &mut [f32],
    output: &mut [f32],
) -> Result<()> {
    let _profile = profile::scope(Category::Qass);
    let embed_dim = checked_elements(heads, head_dim, "QASSMax embed dimension")?;
    let query_rows = checked_elements(heads, rows, "QASSMax query rows")?;
    let query_elements = checked_elements(query_rows, head_dim, "QASSMax query")?;
    require_length(queries, query_elements, "QASSMax query")?;
    require_length(output, query_elements, "QASSMax output")?;
    require_length(weights.base_first_weight, hidden_dim, "QASSMax base first weight")?;
    require_length(weights.base_first_bias, hidden_dim, "QASSMax base first bias")?;
    require_length(
        weights.base_second_weight,
        checked_elements(embed_dim, hidden_dim, "QASSMax base second weight")?,
        "QASSMax base second weight",
    )?;
    require_length(weights.base_second_bias, embed_dim, "QASSMax base second bias")?;
    require_length(
        weights.query_first_weight,
        checked_elements(hidden_dim, head_dim, "QASSMax query first weight")?,
        "QASSMax query first weight",
    )?;
    require_length(weights.query_first_bias, hidden_dim, "QASSMax query first bias")?;
    require_length(
        weights.query_second_weight,
        checked_elements(head_dim, hidden_dim, "QASSMax query second weight")?,
        "QASSMax query second weight",
    )?;
    require_length(weights.query_second_bias, head_dim, "QASSMax query second bias")?;
    require_length(base_hidden_scratch, hidden_dim, "QASSMax base hidden scratch")?;
    require_length(base_scale_scratch, embed_dim, "QASSMax base scale scratch")?;
    require_length(
        query_hidden_scratch,
        checked_elements(query_rows, hidden_dim, "QASSMax query hidden scratch")?,
        "QASSMax query hidden scratch",
    )?;

    let log_source_len = (source_len.max(1) as f32).ln();
    for hidden in 0..hidden_dim {
        base_hidden_scratch[hidden] =
            gelu(weights.base_first_weight[hidden].mul_add(log_source_len, weights.base_first_bias[hidden]));
    }
    affine_f32(
        base_hidden_scratch,
        1,
        hidden_dim,
        weights.base_second_weight,
        Some(weights.base_second_bias),
        embed_dim,
        base_scale_scratch,
    )?;
    affine_f32(
        queries,
        query_rows,
        head_dim,
        weights.query_first_weight,
        Some(weights.query_first_bias),
        hidden_dim,
        query_hidden_scratch,
    )?;
    for value in &mut *query_hidden_scratch {
        *value = gelu(*value);
    }
    affine_f32(
        query_hidden_scratch,
        query_rows,
        hidden_dim,
        weights.query_second_weight,
        Some(weights.query_second_bias),
        head_dim,
        output,
    )?;
    for head in 0..heads {
        for row in 0..rows {
            let offset = (head * rows + row) * head_dim;
            let query = &queries[offset..][..head_dim];
            let output_row = &mut output[offset..][..head_dim];
            for dimension in 0..head_dim {
                output_row[dimension] = query[dimension]
                    * base_scale_scratch[head * head_dim + dimension]
                    * (1.0 + qass_tanh(output_row[dimension]));
            }
        }
    }
    Ok(())
}

/// Apply checkpoint-provided non-interleaved RoPE frequencies to `[heads, rows, head_dim]`.
pub fn rope_non_interleaved_f32(
    input: &[f32],
    heads: usize,
    rows: usize,
    head_dim: usize,
    frequencies: &[f32],
    position_offset: usize,
    output: &mut [f32],
) -> Result<()> {
    let _profile = profile::scope(Category::Rope);
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(Error::InvalidArgument(
            "RoPE head_dim must be a non-zero even number".into(),
        ));
    }
    let elements = checked_elements(
        checked_elements(heads, rows, "RoPE heads and rows")?,
        head_dim,
        "RoPE input",
    )?;
    require_length(input, elements, "RoPE input")?;
    require_length(output, elements, "RoPE output")?;
    require_length(frequencies, head_dim / 2, "RoPE frequencies")?;
    let _ = position_offset
        .checked_add(rows)
        .ok_or(Error::ArithmeticOverflow("RoPE positions"))?;
    let half = head_dim / 2;
    for row in 0..rows {
        let position = (position_offset + row) as f32;
        for dimension in 0..half {
            let angle = position * frequencies[dimension];
            let (sin, cos) = angle.sin_cos();
            for head in 0..heads {
                let offset = (head * rows + row) * head_dim;
                let left = input[offset + dimension];
                let right = input[offset + half + dimension];
                output[offset + dimension] = left * cos - right * sin;
                output[offset + half + dimension] = right * cos + left * sin;
            }
        }
    }
    Ok(())
}

/// Materialize stable scaled dot-product attention for `[heads, rows, head_dim]` tensors.
pub fn scaled_dot_product_attention_f32(
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[f32],
    source_rows: usize,
    heads: usize,
    head_dim: usize,
    score_scratch: &mut [f32],
    output: &mut [f32],
) -> Result<()> {
    let _profile = profile::scope(Category::Attention);
    let query_elements = checked_elements(
        checked_elements(heads, query_rows, "attention query heads and rows")?,
        head_dim,
        "attention query",
    )?;
    let source_elements = checked_elements(
        checked_elements(heads, source_rows, "attention source heads and rows")?,
        head_dim,
        "attention source",
    )?;
    let score_elements = checked_elements(
        checked_elements(heads, query_rows, "attention score heads and rows")?,
        source_rows,
        "attention scores",
    )?;
    require_length(query, query_elements, "attention query")?;
    require_length(key, source_elements, "attention key")?;
    require_length(value, source_elements, "attention value")?;
    require_length(score_scratch, score_elements, "attention score scratch")?;
    require_length(output, query_elements, "attention output")?;
    let scale = (head_dim as f32).sqrt().recip();
    for head in 0..heads {
        for row in 0..query_rows {
            let query_row = &query[(head * query_rows + row) * head_dim..][..head_dim];
            let scores = &mut score_scratch[(head * query_rows + row) * source_rows..][..source_rows];
            for source in 0..source_rows {
                scores[source] = query_row
                    .iter()
                    .zip(&key[(head * source_rows + source) * head_dim..][..head_dim])
                    .fold(0.0, |sum, (q, k)| q.mul_add(*k, sum))
                    * scale;
            }
            softmax_f32(scores)?;
            let output_row = &mut output[(head * query_rows + row) * head_dim..][..head_dim];
            output_row.fill(0.0);
            for source in 0..source_rows {
                for dimension in 0..head_dim {
                    output_row[dimension] = scores[source].mul_add(
                        value[(head * source_rows + source) * head_dim + dimension],
                        output_row[dimension],
                    );
                }
            }
        }
    }
    Ok(())
}

/// Caller-owned scratch for tiled online attention.
///
/// `scores` holds one `[heads, query_block_rows, key_tile_rows]` tile. `maximum`
/// and `denominator` hold one value per query row in that block.
pub struct OnlineAttentionScratch<'a> {
    /// Score storage for one tile.
    pub scores: &'a mut [f32],
    /// Running maximum for each query row.
    pub maximum: &'a mut [f32],
    /// Running softmax denominator for each query row.
    pub denominator: &'a mut [f32],
}

/// Compute stable scaled dot-product attention without materializing all scores.
///
/// Input and output tensors use `[heads, rows, head_dim]` storage. The caller
/// selects bounded query blocks and key tiles and reuses the supplied scratch.
#[allow(clippy::too_many_lines)]
pub fn tiled_online_scaled_dot_product_attention_f32(
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[f32],
    source_rows: usize,
    heads: usize,
    head_dim: usize,
    query_block_rows: usize,
    key_tile_rows: usize,
    scratch: &mut OnlineAttentionScratch<'_>,
    output: &mut [f32],
) -> Result<()> {
    tiled_online_scaled_dot_product_attention_with_backend_f32(
        query,
        query_rows,
        key,
        value,
        source_rows,
        heads,
        head_dim,
        query_block_rows,
        key_tile_rows,
        super::native::attention_backend(),
        scratch,
        output,
    )
}

/// Execute tiled online attention with a previously selected native backend.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn tiled_online_scaled_dot_product_attention_with_backend_f32(
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[f32],
    source_rows: usize,
    heads: usize,
    head_dim: usize,
    query_block_rows: usize,
    key_tile_rows: usize,
    backend: super::native::AttentionBackend,
    scratch: &mut OnlineAttentionScratch<'_>,
    output: &mut [f32],
) -> Result<()> {
    let _profile = profile::scope(Category::Attention);
    let query_elements = checked_elements(
        checked_elements(heads, query_rows, "attention query heads and rows")?,
        head_dim,
        "attention query",
    )?;
    let source_elements = checked_elements(
        checked_elements(heads, source_rows, "attention source heads and rows")?,
        head_dim,
        "attention source",
    )?;
    if query_block_rows == 0 || key_tile_rows == 0 {
        return Err(Error::InvalidArgument(
            "attention tile dimensions must be non-zero".into(),
        ));
    }
    require_length(query, query_elements, "attention query")?;
    require_length(key, source_elements, "attention key")?;
    require_length(value, source_elements, "attention value")?;
    require_length(output, query_elements, "attention output")?;
    let block_rows = query_rows.min(query_block_rows);
    let tile_rows = source_rows.min(key_tile_rows);
    let score_elements = checked_elements(
        checked_elements(heads, block_rows, "attention score heads and rows")?,
        tile_rows,
        "attention score scratch",
    )?;
    let state_elements = checked_elements(heads, block_rows, "attention online state")?;
    if scratch.scores.len() < score_elements {
        return Err(Error::InvalidArgument(format!(
            "attention score scratch has length {}; expected at least {score_elements}",
            scratch.scores.len()
        )));
    }
    if scratch.maximum.len() < state_elements || scratch.denominator.len() < state_elements {
        return Err(Error::InvalidArgument(
            "attention online state scratch is too small".into(),
        ));
    }

    let scale = (head_dim as f32).sqrt().recip();
    for query_start in (0..query_rows).step_by(query_block_rows) {
        let rows = (query_rows - query_start).min(query_block_rows);
        for head in 0..heads {
            let state = head * rows;
            scratch.maximum[state..][..rows].fill(f32::NEG_INFINITY);
            scratch.denominator[state..][..rows].fill(0.0);
            for row in 0..rows {
                output[(head * query_rows + query_start + row) * head_dim..][..head_dim].fill(0.0);
            }
        }
        for source_start in (0..source_rows).step_by(key_tile_rows) {
            let sources = (source_rows - source_start).min(key_tile_rows);
            for head in 0..heads {
                let query_tile_start = (head * query_rows + query_start) * head_dim;
                let query_tile = &query[query_tile_start..][..rows * head_dim];
                let source_tile_start = (head * source_rows + source_start) * head_dim;
                let key_tile = &key[source_tile_start..][..sources * head_dim];
                let score_tile = &mut scratch.scores[..rows * sources];
                if !super::native::attention_score_tile_f32(
                    backend, query_tile, rows, key_tile, sources, head_dim, scale, score_tile,
                ) {
                    for row in 0..rows {
                        let query_row = &query_tile[row * head_dim..][..head_dim];
                        let scores = &mut score_tile[row * sources..][..sources];
                        for source in 0..sources {
                            let key_row = &key_tile[source * head_dim..][..head_dim];
                            let dot = super::native::attention_dot_f32(backend, query_row, key_row)
                                .unwrap_or_else(|| {
                                    query_row
                                        .iter()
                                        .zip(key_row)
                                        .fold(0.0, |sum, (q, k)| q.mul_add(*k, sum))
                                });
                            scores[source] = dot * scale;
                        }
                    }
                }
                for row in 0..rows {
                    let scores = &mut score_tile[row * sources..][..sources];
                    let state = head * rows + row;
                    let tile_maximum = attention_score_maximum(backend, scores)?;
                    let maximum = scratch.maximum[state].max(tile_maximum);
                    if maximum.is_infinite() && maximum.is_sign_positive() {
                        return Err(Error::InvalidArgument(
                            "softmax normalization is not finite".into(),
                        ));
                    }
                    if maximum == f32::NEG_INFINITY {
                        continue;
                    }
                    let output_row =
                        &mut output[(head * query_rows + query_start + row) * head_dim..][..head_dim];
                    if scratch.denominator[state] != 0.0 {
                        let old_scale = (scratch.maximum[state] - maximum).exp();
                        for value in &mut *output_row {
                            *value *= old_scale;
                        }
                        scratch.denominator[state] *= old_scale;
                    }
                    if let Some(sum) = super::native::softmax_exp_sum_f32(backend, scores, maximum) {
                        scratch.denominator[state] += sum;
                    } else {
                        for score in &mut *scores {
                            let weight = (*score - maximum).exp();
                            scratch.denominator[state] += weight;
                            *score = weight;
                        }
                    }
                    scratch.maximum[state] = maximum;
                }
                let value_tile = &value[source_tile_start..][..sources * head_dim];
                let output_tile_start = (head * query_rows + query_start) * head_dim;
                let output_tile = &mut output[output_tile_start..][..rows * head_dim];
                if !super::native::attention_value_tile_fma_f32(
                    backend,
                    score_tile,
                    rows,
                    value_tile,
                    sources,
                    head_dim,
                    output_tile,
                ) {
                    for row in 0..rows {
                        let output_row = &mut output_tile[row * head_dim..][..head_dim];
                        for source in 0..sources {
                            let value_row = &value_tile[source * head_dim..][..head_dim];
                            if !super::native::attention_value_fma_f32(
                                backend,
                                score_tile[row * sources + source],
                                value_row,
                                output_row,
                            ) {
                                for dimension in 0..head_dim {
                                    output_row[dimension] = score_tile[row * sources + source]
                                        .mul_add(value_row[dimension], output_row[dimension]);
                                }
                            }
                        }
                    }
                }
            }
        }
        for head in 0..heads {
            for row in 0..rows {
                let denominator = scratch.denominator[head * rows + row];
                if !denominator.is_finite() || denominator == 0.0 {
                    return Err(Error::InvalidArgument(
                        "softmax normalization is not finite".into(),
                    ));
                }
                for value in &mut output[(head * query_rows + query_start + row) * head_dim..][..head_dim] {
                    *value /= denominator;
                }
            }
        }
    }
    Ok(())
}

/// Execute cached attention with source tiles outermost so one K/V tile serves
/// every bounded query block while it remains cache-resident.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn source_major_online_attention_with_backend_f32(
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[f32],
    source_rows: usize,
    heads: usize,
    head_dim: usize,
    query_block_rows: usize,
    key_tile_rows: usize,
    key_is_packed: bool,
    backend: super::native::AttentionBackend,
    scratch: &mut OnlineAttentionScratch<'_>,
    output: &mut [f32],
) -> Result<()> {
    let _profile = profile::scope(Category::Attention);
    let query_elements = checked_elements(
        checked_elements(heads, query_rows, "attention query heads and rows")?,
        head_dim,
        "attention query",
    )?;
    let source_elements = checked_elements(
        checked_elements(heads, source_rows, "attention source heads and rows")?,
        head_dim,
        "attention source",
    )?;
    if query_block_rows == 0 || key_tile_rows == 0 {
        return Err(Error::InvalidArgument(
            "attention tile dimensions must be non-zero".into(),
        ));
    }
    require_length(query, query_elements, "attention query")?;
    require_length(key, source_elements, "attention key")?;
    require_length(value, source_elements, "attention value")?;
    require_length(output, query_elements, "attention output")?;
    let block_rows = query_rows.min(query_block_rows);
    let tile_rows = source_rows.min(key_tile_rows);
    let packed_tile_rows = tile_rows.div_ceil(16) * 16;
    let score_elements = checked_elements(block_rows, tile_rows, "attention score scratch")?;
    let state_elements = checked_elements(heads, query_rows, "attention online state")?;
    if scratch.scores.len() < score_elements {
        return Err(Error::InvalidArgument(format!(
            "attention score scratch has length {}; expected at least {score_elements}",
            scratch.scores.len()
        )));
    }
    if scratch.maximum.len() < state_elements || scratch.denominator.len() < state_elements {
        return Err(Error::InvalidArgument(
            "attention online state scratch is too small".into(),
        ));
    }

    scratch.maximum[..state_elements].fill(f32::NEG_INFINITY);
    scratch.denominator[..state_elements].fill(0.0);
    output.fill(0.0);
    let scale = (head_dim as f32).sqrt().recip();
    let mut packed_key = vec![0.0; packed_tile_rows * head_dim];
    for source_start in (0..source_rows).step_by(key_tile_rows) {
        let sources = (source_rows - source_start).min(key_tile_rows);
        let packed_sources = sources.div_ceil(16) * 16;
        for head in 0..heads {
            let source_tile_start = (head * source_rows + source_start) * head_dim;
            let key_tile = &key[source_tile_start..][..sources * head_dim];
            let value_tile = &value[source_tile_start..][..sources * head_dim];
            if !key_is_packed {
                packed_key[..packed_sources * head_dim].fill(0.0);
                for dimension in 0..head_dim {
                    for source in 0..sources {
                        packed_key[dimension * packed_sources + source] =
                            key_tile[source * head_dim + dimension];
                    }
                }
            }
            let packed_key_tile = if key_is_packed {
                key_tile
            } else {
                &packed_key[..packed_sources * head_dim]
            };
            for query_start in (0..query_rows).step_by(query_block_rows) {
                let rows = (query_rows - query_start).min(query_block_rows);
                let query_tile_start = (head * query_rows + query_start) * head_dim;
                let query_tile = &query[query_tile_start..][..rows * head_dim];
                let score_tile = &mut scratch.scores[..rows * sources];
                let packed_score = if key_is_packed {
                    super::native::attention_score_packed_key_tile_f32(
                        backend,
                        query_tile,
                        rows,
                        packed_key_tile,
                        sources,
                        head_dim,
                        scale,
                        score_tile,
                    )
                } else {
                    super::native::attention_score_packed_key_padded_tile_f32(
                        backend,
                        query_tile,
                        rows,
                        packed_key_tile,
                        sources,
                        packed_sources,
                        head_dim,
                        scale,
                        score_tile,
                    )
                };
                if !packed_score {
                    if key_is_packed {
                        for row in 0..rows {
                            let query_row = &query_tile[row * head_dim..][..head_dim];
                            let scores = &mut score_tile[row * sources..][..sources];
                            for source in 0..sources {
                                scores[source] = (0..head_dim).fold(0.0, |sum, dimension| {
                                    query_row[dimension].mul_add(key_tile[dimension * sources + source], sum)
                                }) * scale;
                            }
                        }
                    } else if !super::native::attention_score_tile_f32(
                        backend, query_tile, rows, key_tile, sources, head_dim, scale, score_tile,
                    ) {
                        for row in 0..rows {
                            let query_row = &query_tile[row * head_dim..][..head_dim];
                            let scores = &mut score_tile[row * sources..][..sources];
                            for source in 0..sources {
                                let key_row = &key_tile[source * head_dim..][..head_dim];
                                let dot = super::native::attention_dot_f32(backend, query_row, key_row)
                                    .unwrap_or_else(|| {
                                        query_row
                                            .iter()
                                            .zip(key_row)
                                            .fold(0.0, |sum, (q, k)| q.mul_add(*k, sum))
                                    });
                                scores[source] = dot * scale;
                            }
                        }
                    }
                }
                for row in 0..rows {
                    let scores = &mut score_tile[row * sources..][..sources];
                    let state = head * query_rows + query_start + row;
                    let tile_maximum = attention_score_maximum(backend, scores)?;
                    let maximum = scratch.maximum[state].max(tile_maximum);
                    if maximum.is_infinite() && maximum.is_sign_positive() {
                        return Err(Error::InvalidArgument(
                            "softmax normalization is not finite".into(),
                        ));
                    }
                    if maximum == f32::NEG_INFINITY {
                        continue;
                    }
                    let output_row =
                        &mut output[(head * query_rows + query_start + row) * head_dim..][..head_dim];
                    if scratch.denominator[state] != 0.0 {
                        let old_scale = (scratch.maximum[state] - maximum).exp();
                        for result in &mut *output_row {
                            *result *= old_scale;
                        }
                        scratch.denominator[state] *= old_scale;
                    }
                    if let Some(sum) = super::native::softmax_exp_sum_f32(backend, scores, maximum) {
                        scratch.denominator[state] += sum;
                    } else {
                        for score in &mut *scores {
                            let weight = (*score - maximum).exp();
                            scratch.denominator[state] += weight;
                            *score = weight;
                        }
                    }
                    scratch.maximum[state] = maximum;
                }
                let output_tile_start = (head * query_rows + query_start) * head_dim;
                let output_tile = &mut output[output_tile_start..][..rows * head_dim];
                if !super::native::attention_value_tile_fma_f32(
                    backend,
                    score_tile,
                    rows,
                    value_tile,
                    sources,
                    head_dim,
                    output_tile,
                ) {
                    for row in 0..rows {
                        let output_row = &mut output_tile[row * head_dim..][..head_dim];
                        for source in 0..sources {
                            let value_row = &value_tile[source * head_dim..][..head_dim];
                            if !super::native::attention_value_fma_f32(
                                backend,
                                score_tile[row * sources + source],
                                value_row,
                                output_row,
                            ) {
                                for dimension in 0..head_dim {
                                    output_row[dimension] = score_tile[row * sources + source]
                                        .mul_add(value_row[dimension], output_row[dimension]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    for head in 0..heads {
        for row in 0..query_rows {
            let denominator = scratch.denominator[head * query_rows + row];
            if !denominator.is_finite() || denominator == 0.0 {
                return Err(Error::InvalidArgument(
                    "softmax normalization is not finite".into(),
                ));
            }
            for result in &mut output[(head * query_rows + row) * head_dim..][..head_dim] {
                *result /= denominator;
            }
        }
    }
    Ok(())
}

/// Execute source-major cached attention from BF16 K/V storage with FP32
/// score, softmax, and value accumulation.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn source_major_online_attention_bf16_with_backend_f32(
    query: &[f32],
    query_rows: usize,
    key: &[u16],
    value: &[u16],
    source_rows: usize,
    heads: usize,
    head_dim: usize,
    query_block_rows: usize,
    key_tile_rows: usize,
    key_is_packed: bool,
    backend: super::native::AttentionBackend,
    scratch: &mut OnlineAttentionScratch<'_>,
    output: &mut [f32],
) -> Result<()> {
    let _profile = profile::scope(Category::Attention);
    let query_elements = checked_elements(
        checked_elements(heads, query_rows, "attention query heads and rows")?,
        head_dim,
        "attention query",
    )?;
    let source_elements = checked_elements(
        checked_elements(heads, source_rows, "attention source heads and rows")?,
        head_dim,
        "attention source",
    )?;
    if query_block_rows == 0 || key_tile_rows == 0 {
        return Err(Error::InvalidArgument(
            "attention tile dimensions must be non-zero".into(),
        ));
    }
    require_length(query, query_elements, "attention query")?;
    if key.len() != source_elements || value.len() != source_elements {
        return Err(Error::InvalidArgument(
            "BF16 attention K/V length does not match dimensions".into(),
        ));
    }
    require_length(output, query_elements, "attention output")?;
    let block_rows = query_rows.min(query_block_rows);
    let tile_rows = source_rows.min(key_tile_rows);
    let score_elements = checked_elements(block_rows, tile_rows, "attention score scratch")?;
    let state_elements = checked_elements(heads, query_rows, "attention online state")?;
    if scratch.scores.len() < score_elements {
        return Err(Error::InvalidArgument(format!(
            "attention score scratch has length {}; expected at least {score_elements}",
            scratch.scores.len()
        )));
    }
    if scratch.maximum.len() < state_elements || scratch.denominator.len() < state_elements {
        return Err(Error::InvalidArgument(
            "attention online state scratch is too small".into(),
        ));
    }

    scratch.maximum[..state_elements].fill(f32::NEG_INFINITY);
    scratch.denominator[..state_elements].fill(0.0);
    output.fill(0.0);
    let scale = (head_dim as f32).sqrt().recip();
    let mut packed_key = vec![0_u16; tile_rows * head_dim];
    for source_start in (0..source_rows).step_by(key_tile_rows) {
        let sources = (source_rows - source_start).min(key_tile_rows);
        let tile_elements = sources * head_dim;
        for head in 0..heads {
            let source_tile_start = (head * source_rows + source_start) * head_dim;
            let key_tile = &key[source_tile_start..][..tile_elements];
            let value_tile = &value[source_tile_start..][..tile_elements];
            if !key_is_packed {
                for dimension in 0..head_dim {
                    for source in 0..sources {
                        packed_key[dimension * sources + source] = key_tile[source * head_dim + dimension];
                    }
                }
            }
            let packed_key_tile = if key_is_packed {
                key_tile
            } else {
                &packed_key[..tile_elements]
            };
            for query_start in (0..query_rows).step_by(query_block_rows) {
                let rows = (query_rows - query_start).min(query_block_rows);
                let query_tile_start = (head * query_rows + query_start) * head_dim;
                let query_tile = &query[query_tile_start..][..rows * head_dim];
                let score_tile = &mut scratch.scores[..rows * sources];
                let packed_score = super::native::attention_score_packed_key_bf16_tile_f32(
                    backend,
                    query_tile,
                    rows,
                    packed_key_tile,
                    sources,
                    head_dim,
                    scale,
                    score_tile,
                );
                if !packed_score {
                    if key_is_packed {
                        for row in 0..rows {
                            let query_row = &query_tile[row * head_dim..][..head_dim];
                            let scores = &mut score_tile[row * sources..][..sources];
                            for source in 0..sources {
                                scores[source] = (0..head_dim).fold(0.0, |sum, dimension| {
                                    query_row[dimension].mul_add(
                                        f32::from_bits(
                                            u32::from(key_tile[dimension * sources + source]) << 16,
                                        ),
                                        sum,
                                    )
                                }) * scale;
                            }
                        }
                    } else if !super::native::attention_score_tile_bf16_f32(
                        backend, query_tile, rows, key_tile, sources, head_dim, scale, score_tile,
                    ) {
                        for row in 0..rows {
                            let query_row = &query_tile[row * head_dim..][..head_dim];
                            let scores = &mut score_tile[row * sources..][..sources];
                            for source in 0..sources {
                                let key_row = &key_tile[source * head_dim..][..head_dim];
                                let dot = query_row.iter().zip(key_row).fold(0.0, |sum, (q, k)| {
                                    q.mul_add(f32::from_bits(u32::from(*k) << 16), sum)
                                });
                                scores[source] = dot * scale;
                            }
                        }
                    }
                }
                for row in 0..rows {
                    let scores = &mut score_tile[row * sources..][..sources];
                    let state = head * query_rows + query_start + row;
                    let tile_maximum = attention_score_maximum(backend, scores)?;
                    let maximum = scratch.maximum[state].max(tile_maximum);
                    if maximum.is_infinite() && maximum.is_sign_positive() {
                        return Err(Error::InvalidArgument(
                            "softmax normalization is not finite".into(),
                        ));
                    }
                    if maximum == f32::NEG_INFINITY {
                        continue;
                    }
                    let output_row =
                        &mut output[(head * query_rows + query_start + row) * head_dim..][..head_dim];
                    if scratch.denominator[state] != 0.0 {
                        let old_scale = (scratch.maximum[state] - maximum).exp();
                        for result in &mut *output_row {
                            *result *= old_scale;
                        }
                        scratch.denominator[state] *= old_scale;
                    }
                    if let Some(sum) = super::native::softmax_exp_sum_f32(backend, scores, maximum) {
                        scratch.denominator[state] += sum;
                    } else {
                        for score in &mut *scores {
                            let weight = (*score - maximum).exp();
                            scratch.denominator[state] += weight;
                            *score = weight;
                        }
                    }
                    scratch.maximum[state] = maximum;
                }
                let output_tile_start = (head * query_rows + query_start) * head_dim;
                let output_tile = &mut output[output_tile_start..][..rows * head_dim];
                if !super::native::attention_value_tile_bf16_fma_f32(
                    backend,
                    score_tile,
                    rows,
                    value_tile,
                    sources,
                    head_dim,
                    output_tile,
                ) {
                    for row in 0..rows {
                        let output_row = &mut output_tile[row * head_dim..][..head_dim];
                        for source in 0..sources {
                            let value_row = &value_tile[source * head_dim..][..head_dim];
                            for dimension in 0..head_dim {
                                output_row[dimension] = score_tile[row * sources + source].mul_add(
                                    f32::from_bits(u32::from(value_row[dimension]) << 16),
                                    output_row[dimension],
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    for head in 0..heads {
        for row in 0..query_rows {
            let denominator = scratch.denominator[head * query_rows + row];
            if !denominator.is_finite() || denominator == 0.0 {
                return Err(Error::InvalidArgument(
                    "softmax normalization is not finite".into(),
                ));
            }
            for result in &mut output[(head * query_rows + row) * head_dim..][..head_dim] {
                *result /= denominator;
            }
        }
    }
    Ok(())
}

/// Execute source-major cached attention with FP32 K, BF16 V, and FP32
/// arithmetic. This policy preserves score precision while reducing V traffic.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub(crate) fn source_major_online_attention_f32_key_bf16_value_with_backend_f32(
    query: &[f32],
    query_rows: usize,
    key: &[f32],
    value: &[u16],
    source_rows: usize,
    heads: usize,
    head_dim: usize,
    query_block_rows: usize,
    key_tile_rows: usize,
    key_is_packed: bool,
    backend: super::native::AttentionBackend,
    scratch: &mut OnlineAttentionScratch<'_>,
    output: &mut [f32],
) -> Result<()> {
    let _profile = profile::scope(Category::Attention);
    let query_elements = checked_elements(
        checked_elements(heads, query_rows, "attention query heads and rows")?,
        head_dim,
        "attention query",
    )?;
    let source_elements = checked_elements(
        checked_elements(heads, source_rows, "attention source heads and rows")?,
        head_dim,
        "attention source",
    )?;
    if query_block_rows == 0 || key_tile_rows == 0 {
        return Err(Error::InvalidArgument(
            "attention tile dimensions must be non-zero".into(),
        ));
    }
    require_length(query, query_elements, "attention query")?;
    require_length(key, source_elements, "attention key")?;
    if value.len() != source_elements {
        return Err(Error::InvalidArgument(
            "BF16 attention V length does not match dimensions".into(),
        ));
    }
    require_length(output, query_elements, "attention output")?;
    let block_rows = query_rows.min(query_block_rows);
    let tile_rows = source_rows.min(key_tile_rows);
    let score_elements = checked_elements(block_rows, tile_rows, "attention score scratch")?;
    let state_elements = checked_elements(heads, query_rows, "attention online state")?;
    if scratch.scores.len() < score_elements {
        return Err(Error::InvalidArgument(format!(
            "attention score scratch has length {}; expected at least {score_elements}",
            scratch.scores.len()
        )));
    }
    if scratch.maximum.len() < state_elements || scratch.denominator.len() < state_elements {
        return Err(Error::InvalidArgument(
            "attention online state scratch is too small".into(),
        ));
    }

    scratch.maximum[..state_elements].fill(f32::NEG_INFINITY);
    scratch.denominator[..state_elements].fill(0.0);
    output.fill(0.0);
    let scale = (head_dim as f32).sqrt().recip();
    let mut packed_key = vec![0.0; tile_rows * head_dim];
    let use_bf16_value_dot = bf16_attention_value_dot_enabled();
    let mut packed_value = use_bf16_value_dot.then(|| vec![0_u16; tile_rows * head_dim]);
    let mut weights_bf16 = use_bf16_value_dot.then(|| vec![0_u16; block_rows * tile_rows]);
    let mut profile_scores =
        (attention_stats_enabled() && source_rows >= 4096).then(|| vec![0.0; source_rows]);
    for source_start in (0..source_rows).step_by(key_tile_rows) {
        let sources = (source_rows - source_start).min(key_tile_rows);
        let tile_elements = sources * head_dim;
        for head in 0..heads {
            let source_tile_start = (head * source_rows + source_start) * head_dim;
            let key_tile = &key[source_tile_start..][..tile_elements];
            let value_tile = &value[source_tile_start..][..tile_elements];
            let value_is_packed = packed_value.as_mut().is_some_and(|packed| {
                super::native::pack_attention_value_bf16(
                    value_tile,
                    sources,
                    head_dim,
                    &mut packed[..tile_elements],
                )
            });
            if !key_is_packed {
                for dimension in 0..head_dim {
                    for source in 0..sources {
                        packed_key[dimension * sources + source] = key_tile[source * head_dim + dimension];
                    }
                }
            }
            let packed_key_tile = if key_is_packed {
                key_tile
            } else {
                &packed_key[..sources * head_dim]
            };
            for query_start in (0..query_rows).step_by(query_block_rows) {
                let rows = (query_rows - query_start).min(query_block_rows);
                let query_tile_start = (head * query_rows + query_start) * head_dim;
                let query_tile = &query[query_tile_start..][..rows * head_dim];
                let score_tile = &mut scratch.scores[..rows * sources];
                let packed_score = {
                    let _profile = profile::scope(Category::AttentionScore);
                    super::native::attention_score_packed_key_tile_f32(
                        backend,
                        query_tile,
                        rows,
                        packed_key_tile,
                        sources,
                        head_dim,
                        scale,
                        score_tile,
                    )
                };
                if !packed_score {
                    if key_is_packed {
                        for row in 0..rows {
                            let query_row = &query_tile[row * head_dim..][..head_dim];
                            let scores = &mut score_tile[row * sources..][..sources];
                            for source in 0..sources {
                                scores[source] = (0..head_dim).fold(0.0, |sum, dimension| {
                                    query_row[dimension].mul_add(key_tile[dimension * sources + source], sum)
                                }) * scale;
                            }
                        }
                    } else if !super::native::attention_score_tile_f32(
                        backend, query_tile, rows, key_tile, sources, head_dim, scale, score_tile,
                    ) {
                        for row in 0..rows {
                            let query_row = &query_tile[row * head_dim..][..head_dim];
                            let scores = &mut score_tile[row * sources..][..sources];
                            for source in 0..sources {
                                let key_row = &key_tile[source * head_dim..][..head_dim];
                                let dot = super::native::attention_dot_f32(backend, query_row, key_row)
                                    .unwrap_or_else(|| {
                                        query_row
                                            .iter()
                                            .zip(key_row)
                                            .fold(0.0, |sum, (q, k)| q.mul_add(*k, sum))
                                    });
                                scores[source] = dot * scale;
                            }
                        }
                    }
                }
                if query_start == 0
                    && let Some(profile) = &mut profile_scores
                {
                    profile[source_start..][..sources].copy_from_slice(&score_tile[..sources]);
                }
                {
                    let _profile = profile::scope(Category::AttentionSoftmax);
                    for row in 0..rows {
                        let scores = &mut score_tile[row * sources..][..sources];
                        let state = head * query_rows + query_start + row;
                        let tile_maximum = attention_score_maximum(backend, scores)?;
                        let maximum = scratch.maximum[state].max(tile_maximum);
                        if maximum.is_infinite() && maximum.is_sign_positive() {
                            return Err(Error::InvalidArgument(
                                "softmax normalization is not finite".into(),
                            ));
                        }
                        if maximum == f32::NEG_INFINITY {
                            continue;
                        }
                        let output_row =
                            &mut output[(head * query_rows + query_start + row) * head_dim..][..head_dim];
                        if scratch.denominator[state] != 0.0 {
                            let old_scale = (scratch.maximum[state] - maximum).exp();
                            for result in &mut *output_row {
                                *result *= old_scale;
                            }
                            scratch.denominator[state] *= old_scale;
                        }
                        if let Some(sum) = super::native::softmax_exp_sum_f32(backend, scores, maximum) {
                            scratch.denominator[state] += sum;
                        } else {
                            for score in &mut *scores {
                                let weight = (*score - maximum).exp();
                                scratch.denominator[state] += weight;
                                *score = weight;
                            }
                        }
                        scratch.maximum[state] = maximum;
                    }
                }
                let output_tile_start = (head * query_rows + query_start) * head_dim;
                let output_tile = &mut output[output_tile_start..][..rows * head_dim];
                {
                    let _profile = profile::scope(Category::AttentionValue);
                    let used_bf16_dot = value_is_packed
                        && super::native::attention_value_packed_bf16_dot_f32(
                            backend,
                            score_tile,
                            rows,
                            &packed_value.as_deref().expect("packed V scratch exists")[..tile_elements],
                            sources,
                            head_dim,
                            output_tile,
                            &mut weights_bf16.as_deref_mut().expect("BF16 weight scratch exists")
                                [..rows * sources],
                        );
                    if !used_bf16_dot
                        && !super::native::attention_value_tile_bf16_fma_f32(
                            backend,
                            score_tile,
                            rows,
                            value_tile,
                            sources,
                            head_dim,
                            output_tile,
                        )
                    {
                        for row in 0..rows {
                            let output_row = &mut output_tile[row * head_dim..][..head_dim];
                            for source in 0..sources {
                                let value_row = &value_tile[source * head_dim..][..head_dim];
                                for dimension in 0..head_dim {
                                    output_row[dimension] = score_tile[row * sources + source].mul_add(
                                        f32::from_bits(u32::from(value_row[dimension]) << 16),
                                        output_row[dimension],
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    {
        let _profile = profile::scope(Category::AttentionFinalize);
        for head in 0..heads {
            for row in 0..query_rows {
                let denominator = scratch.denominator[head * query_rows + row];
                if !denominator.is_finite() || denominator == 0.0 {
                    return Err(Error::InvalidArgument(
                        "softmax normalization is not finite".into(),
                    ));
                }
                for result in &mut output[(head * query_rows + row) * head_dim..][..head_dim] {
                    *result /= denominator;
                }
            }
        }
    }
    if let Some(scores) = profile_scores {
        report_attention_stats(scores);
    }
    Ok(())
}

fn attention_score_maximum(backend: super::native::AttentionBackend, scores: &[f32]) -> Result<f32> {
    if let Some((maximum, has_nan)) = super::native::attention_score_maximum_f32(backend, scores) {
        if has_nan {
            return Err(Error::InvalidArgument("softmax input contains NaN".into()));
        }
        return Ok(maximum);
    }
    let mut maximum = f32::NEG_INFINITY;
    for score in scores {
        if score.is_nan() {
            return Err(Error::InvalidArgument("softmax input contains NaN".into()));
        }
        maximum = maximum.max(*score);
    }
    Ok(maximum)
}

fn bf16_attention_value_dot_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("IRONICL_BF16_VALUE_DOT").is_some())
}

fn attention_stats_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("IRONICL_ATTENTION_STATS").is_some())
}

fn report_attention_stats(mut scores: Vec<f32>) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CALL: AtomicUsize = AtomicUsize::new(0);
    scores.sort_unstable_by(|left, right| right.total_cmp(left));
    let maximum = scores[0];
    let weights = scores
        .iter()
        .map(|score| (*score - maximum).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    let masses = [64_usize, 256, 1024, 4096, 16_384]
        .map(|count| weights[..count.min(weights.len())].iter().sum::<f32>() / total);
    eprintln!(
        "attention_stats call={} rows={} top64={:.6} top256={:.6} top1024={:.6} top4096={:.6} top16384={:.6}",
        CALL.fetch_add(1, Ordering::Relaxed),
        scores.len(),
        masses[0],
        masses[1],
        masses[2],
        masses[3],
        masses[4]
    );
}

/// Convert `[rows, embed_dim]` tokens to `[heads, rows, head_dim]` storage.
pub fn tokens_to_heads_f32(
    input: &[f32],
    rows: usize,
    embed_dim: usize,
    heads: usize,
    output: &mut [f32],
) -> Result<()> {
    let elements = checked_elements(rows, embed_dim, "token-to-head layout")?;
    if heads == 0 || embed_dim % heads != 0 {
        return Err(Error::InvalidArgument(
            "embed_dim must be divisible by a non-zero head count".into(),
        ));
    }
    require_length(input, elements, "token input")?;
    require_length(output, elements, "head output")?;
    let head_dim = embed_dim / heads;
    for row in 0..rows {
        for head in 0..heads {
            output[(head * rows + row) * head_dim..][..head_dim]
                .copy_from_slice(&input[row * embed_dim + head * head_dim..][..head_dim]);
        }
    }
    Ok(())
}

/// Convert `[heads, rows, head_dim]` storage back to `[rows, embed_dim]` tokens.
pub fn heads_to_tokens_f32(
    input: &[f32],
    heads: usize,
    rows: usize,
    head_dim: usize,
    output: &mut [f32],
) -> Result<()> {
    let embed_dim = checked_elements(heads, head_dim, "head merge width")?;
    let elements = checked_elements(rows, embed_dim, "head-to-token layout")?;
    require_length(input, elements, "head input")?;
    require_length(output, elements, "token output")?;
    for row in 0..rows {
        for head in 0..heads {
            output[row * embed_dim + head * head_dim..][..head_dim]
                .copy_from_slice(&input[(head * rows + row) * head_dim..][..head_dim]);
        }
    }
    Ok(())
}

/// Stable FP32 softmax. Optimized implementations must preserve this contract.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] for empty input, NaN input, or a
/// non-finite normalization result.
pub fn softmax_f32(values: &mut [f32]) -> Result<()> {
    if values.is_empty() {
        return Err(Error::InvalidArgument("softmax input must not be empty".into()));
    }
    if values.iter().any(|value| value.is_nan()) {
        return Err(Error::InvalidArgument("softmax input contains NaN".into()));
    }

    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in &mut *values {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    if !sum.is_finite() || sum == 0.0 {
        return Err(Error::InvalidArgument(
            "softmax normalization is not finite".into(),
        ));
    }
    for value in values {
        *value /= sum;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dot_product_uses_matching_inputs() {
        let result = dot_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).expect("valid dot product");
        assert!((result - 32.0).abs() < f32::EPSILON);
    }

    #[test]
    fn softmax_is_stable_for_large_logits() {
        let mut values = [1_000.0, 1_001.0, 1_002.0];
        softmax_f32(&mut values).expect("valid softmax");
        let sum: f32 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(values[2] > values[1] && values[1] > values[0]);
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn affine_matches_pytorch_output_by_input_weight_layout() {
        let input = [1.0, 2.0, -1.0, 3.0];
        let weight = [2.0, -1.0, 0.5, 1.0, 3.0, -2.0];
        let bias = [0.5, -1.0, 2.0];
        let mut output = [0.0; 6];
        affine_f32(&input, 2, 2, &weight, Some(&bias), 3, &mut output).expect("valid affine");
        assert_close(&output, &[0.5, 1.5, 1.0, -4.5, 1.5, -7.0]);
    }

    #[test]
    fn transposed_input3_affine_matches_standard_layout() {
        let input = [1.0, 2.0, -1.0, 3.0, -2.0, 0.5];
        let weight = [2.0, -1.0, 0.5, 1.0, 3.0, -2.0, -0.5, 0.25, 4.0, 2.5, -3.0, 1.5];
        let bias = [0.5, -1.0, 2.0, -0.25];
        let mut transposed = [0.0; 12];
        for output in 0..4 {
            for input in 0..3 {
                transposed[input * 4 + output] = weight[output * 3 + input];
            }
        }
        let mut expected = [0.0; 8];
        affine_f32(&input, 2, 3, &weight, Some(&bias), 4, &mut expected).expect("standard affine");
        let mut actual = [0.0; 8];
        affine_input3_transposed_f32(&input, 2, &transposed, &bias, 4, &mut actual)
            .expect("transposed affine");
        assert_eq!(actual, expected);
    }

    #[test]
    fn embedding_lookup_copies_selected_table_rows() {
        let table = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut output = [0.0; 4];
        embedding_lookup_f32(&[2, 0], &table, 3, 2, &mut output).expect("valid lookup");
        assert_eq!(output, [5.0, 6.0, 1.0, 2.0]);
        assert!(embedding_lookup_f32(&[3], &table, 3, 2, &mut [0.0; 2]).is_err());
    }

    #[test]
    fn one_hot_linear_selects_weight_columns_and_adds_bias() {
        let mut output = [0.0; 4];
        one_hot_linear_f32(
            &[2, 0],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[10.0, 20.0],
            3,
            2,
            &mut output,
        )
        .expect("valid one-hot linear");
        assert_eq!(output, [13.0, 26.0, 11.0, 24.0]);
    }

    #[test]
    fn layer_norm_uses_epsilon_and_optional_bias() {
        let mut output = [0.0; 2];
        layer_norm_f32(&[1.0, 3.0], 1, 2, &[2.0, 3.0], Some(&[1.0, -1.0]), &mut output)
            .expect("valid layer norm");
        let inverse_std = (1.0_f32 + LAYER_NORM_EPSILON).sqrt().recip();
        assert_close(&output, &[1.0 - 2.0 * inverse_std, -1.0 + 3.0 * inverse_std]);
        layer_norm_f32(&[4.0, 4.0], 1, 2, &[1.0, 1.0], None, &mut output).expect("constant row");
        assert_eq!(output, [0.0, 0.0]);
    }

    #[test]
    fn gelu_tracks_erf_form() {
        let mut output = [0.0; 3];
        gelu_f32(&[-1.0, 0.0, 1.0], &mut output).expect("valid GELU");
        assert_close(&output, &[-0.158_655_26, 0.0, 0.841_344_7]);
    }

    #[test]
    fn residual_rejects_mismatched_lengths() {
        let mut output = [0.0; 2];
        residual_add_f32(&[1.0, 2.0], &[3.0, 4.0], &mut output).expect("valid residual");
        assert_eq!(output, [4.0, 6.0]);
        assert!(residual_add_f32(&[1.0], &[2.0, 3.0], &mut output).is_err());
    }

    #[test]
    fn packed_qkv_splits_pytorch_packed_projection() {
        let input = [2.0, 3.0];
        let weight = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, -1.0];
        let bias = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut packed = [0.0; 6];
        let mut query = [0.0; 2];
        let mut key = [0.0; 2];
        let mut value = [0.0; 2];
        packed_qkv_projection_f32(
            &input,
            1,
            2,
            &weight,
            Some(&bias),
            &mut packed,
            &mut query,
            &mut key,
            &mut value,
        )
        .expect("valid QKV");
        assert_eq!(query, [2.0, 4.0]);
        assert_eq!(key, [7.0, 7.0]);
        assert_eq!(value, [10.0, 4.0]);
    }

    #[test]
    fn head_layout_round_trip_preserves_tokens() {
        let tokens = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut heads = [0.0; 8];
        let mut output = [0.0; 8];
        tokens_to_heads_f32(&tokens, 2, 4, 2, &mut heads).expect("valid layout");
        assert_eq!(heads, [1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
        heads_to_tokens_f32(&heads, 2, 2, 2, &mut output).expect("valid merge");
        assert_eq!(output, tokens);
        assert!(tokens_to_heads_f32(&tokens, 2, 4, 3, &mut heads).is_err());
    }

    #[test]
    fn qassmax_applies_base_scale_and_query_modulation() {
        let weights = QassMaxWeights {
            base_first_weight: &[1.0],
            base_first_bias: &[0.0],
            base_second_weight: &[2.0, 3.0],
            base_second_bias: &[0.0, 0.0],
            query_first_weight: &[1.0, 0.0],
            query_first_bias: &[0.0],
            query_second_weight: &[1.0, 0.0],
            query_second_bias: &[0.0, 0.0],
        };
        let mut base_hidden = [0.0; 1];
        let mut base_scale = [0.0; 2];
        let mut query_hidden = [0.0; 1];
        let mut output = [0.0; 2];
        qassmax_f32(
            &[1.0, 2.0],
            1,
            1,
            2,
            1,
            1,
            weights,
            &mut base_hidden,
            &mut base_scale,
            &mut query_hidden,
            &mut output,
        )
        .expect("valid QASSMax");
        // log(1) is zero, GELU(0) is zero, so the learned base scale is zero.
        assert_eq!(output, [0.0, 0.0]);
        qassmax_f32(
            &[1.0, 2.0],
            1,
            1,
            2,
            2,
            1,
            weights,
            &mut base_hidden,
            &mut base_scale,
            &mut query_hidden,
            &mut output,
        )
        .expect("valid QASSMax");
        assert!(output[0] > 0.0 && output[1] > output[0]);
    }

    #[test]
    fn non_interleaved_rope_rotates_contiguous_halves() {
        let mut output = [0.0; 4];
        rope_non_interleaved_f32(
            &[1.0, 2.0, 3.0, 4.0],
            1,
            1,
            4,
            &[core::f32::consts::FRAC_PI_2, 0.0],
            1,
            &mut output,
        )
        .expect("valid RoPE");
        assert_close(&output, &[-3.0, 2.0, 1.0, 4.0]);
        assert!(rope_non_interleaved_f32(&[1.0; 3], 1, 1, 3, &[1.0], 0, &mut [0.0; 3]).is_err());
    }

    #[test]
    fn attention_materializes_stable_scaled_dot_product() {
        let query = [1.0, 0.0];
        let key = [1.0, 0.0, 0.0, 1.0];
        let value = [10.0, 20.0, 30.0, 40.0];
        let mut scores = [0.0; 2];
        let mut output = [0.0; 2];
        scaled_dot_product_attention_f32(&query, 1, &key, &value, 2, 1, 2, &mut scores, &mut output)
            .expect("valid attention");
        let first = (1.0_f32 / 2.0_f32.sqrt()).exp();
        let weight = first / (first + 1.0);
        assert_close(
            &output,
            &[
                10.0 * weight + 30.0 * (1.0 - weight),
                20.0 * weight + 40.0 * (1.0 - weight),
            ],
        );
        assert!((scores[0] + scores[1] - 1.0).abs() < 1e-6);
        assert!(
            scaled_dot_product_attention_f32(&query, 1, &key, &value, 0, 1, 2, &mut [], &mut output).is_err()
        );
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn tiled_online_attention_matches_materialized_across_tiles_and_scores() {
        let cases = [
            (1, 1, 1, 1, 1, 1),
            (2, 3, 5, 2, 3, 3),
            (3, 9, 17, 4, 7, 7),
            (2, 10, 129, 8, 8, 128),
        ];
        let mut state = 1_u32;
        let mut maximum_error = 0.0_f32;
        for (heads, query_rows, source_rows, head_dim, query_block_rows, key_tile_rows) in cases {
            let elements = |rows| heads * rows * head_dim;
            let mut query = Vec::with_capacity(elements(query_rows));
            let mut key = Vec::with_capacity(elements(source_rows));
            let mut value = Vec::with_capacity(elements(source_rows));
            for values in [&mut query, &mut key, &mut value] {
                for _ in 0..values.capacity() {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    values.push((state as f32 / u32::MAX as f32 - 0.5) * 12.0);
                }
            }
            let mut expected = vec![0.0; elements(query_rows)];
            let mut materialized = vec![0.0; heads * query_rows * source_rows];
            scaled_dot_product_attention_f32(
                &query,
                query_rows,
                &key,
                &value,
                source_rows,
                heads,
                head_dim,
                &mut materialized,
                &mut expected,
            )
            .expect("valid materialized attention");
            let block_rows = query_rows.min(query_block_rows);
            let tile_rows = source_rows.min(key_tile_rows);
            let mut actual = vec![0.0; elements(query_rows)];
            let mut scores = vec![0.0; heads * block_rows * tile_rows];
            let mut maximum = vec![0.0; heads * block_rows];
            let mut denominator = vec![0.0; heads * block_rows];
            tiled_online_scaled_dot_product_attention_f32(
                &query,
                query_rows,
                &key,
                &value,
                source_rows,
                heads,
                head_dim,
                query_block_rows,
                key_tile_rows,
                &mut OnlineAttentionScratch {
                    scores: &mut scores,
                    maximum: &mut maximum,
                    denominator: &mut denominator,
                },
                &mut actual,
            )
            .expect("valid tiled attention");
            maximum_error = maximum_error.max(
                actual
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (actual - expected).abs())
                    .fold(0.0, f32::max),
            );
            assert_close(&actual, &expected);

            let mut source_major = vec![0.0; elements(query_rows)];
            let mut source_major_scores = vec![0.0; block_rows * tile_rows];
            let mut source_major_maximum = vec![0.0; heads * query_rows];
            let mut source_major_denominator = vec![0.0; heads * query_rows];
            source_major_online_attention_with_backend_f32(
                &query,
                query_rows,
                &key,
                &value,
                source_rows,
                heads,
                head_dim,
                query_block_rows,
                key_tile_rows,
                false,
                crate::kernels::native::attention_backend(),
                &mut OnlineAttentionScratch {
                    scores: &mut source_major_scores,
                    maximum: &mut source_major_maximum,
                    denominator: &mut source_major_denominator,
                },
                &mut source_major,
            )
            .expect("valid source-major attention");
            assert_close(&source_major, &expected);

            let mut packed_key = vec![0.0; key.len()];
            for head in 0..heads {
                for source_start in (0..source_rows).step_by(key_tile_rows) {
                    let sources = (source_rows - source_start).min(key_tile_rows);
                    let base = (head * source_rows + source_start) * head_dim;
                    for dimension in 0..head_dim {
                        for source in 0..sources {
                            packed_key[base + dimension * sources + source] =
                                key[(head * source_rows + source_start + source) * head_dim + dimension];
                        }
                    }
                }
            }
            let mut packed_actual = vec![0.0; elements(query_rows)];
            source_major_online_attention_with_backend_f32(
                &query,
                query_rows,
                &packed_key,
                &value,
                source_rows,
                heads,
                head_dim,
                query_block_rows,
                key_tile_rows,
                true,
                crate::kernels::native::attention_backend(),
                &mut OnlineAttentionScratch {
                    scores: &mut source_major_scores,
                    maximum: &mut source_major_maximum,
                    denominator: &mut source_major_denominator,
                },
                &mut packed_actual,
            )
            .expect("valid prepacked source-major attention");
            assert_close(&packed_actual, &expected);
        }
        eprintln!("online attention maximum absolute error: {maximum_error}");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn bf16_source_major_attention_matches_dequantized_fp32() {
        let (heads, query_rows, source_rows, head_dim) = (2, 10, 129, 8);
        let mut state = 7_u32;
        let mut values = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state as f32 / u32::MAX as f32 - 0.5) * 4.0
        };
        let query = (0..heads * query_rows * head_dim)
            .map(|_| values())
            .collect::<Vec<_>>();
        let fp32_key = (0..heads * source_rows * head_dim)
            .map(|_| values())
            .collect::<Vec<_>>();
        let fp32_value = (0..heads * source_rows * head_dim)
            .map(|_| values())
            .collect::<Vec<_>>();
        let to_bf16 = |value: f32| {
            let bits = value.to_bits();
            u16::try_from((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).expect("upper BF16 bits")
        };
        let key = fp32_key.iter().copied().map(to_bf16).collect::<Vec<_>>();
        let value = fp32_value.iter().copied().map(to_bf16).collect::<Vec<_>>();
        let dequantize = |values: &[u16]| {
            values
                .iter()
                .map(|value| f32::from_bits(u32::from(*value) << 16))
                .collect::<Vec<_>>()
        };
        let mut expected = vec![0.0; query.len()];
        let mut materialized = vec![0.0; heads * query_rows * source_rows];
        scaled_dot_product_attention_f32(
            &query,
            query_rows,
            &dequantize(&key),
            &dequantize(&value),
            source_rows,
            heads,
            head_dim,
            &mut materialized,
            &mut expected,
        )
        .expect("valid materialized BF16 attention");
        let mut actual = vec![0.0; query.len()];
        let mut scores = vec![0.0; 8 * 128];
        let mut maximum = vec![0.0; heads * query_rows];
        let mut denominator = vec![0.0; heads * query_rows];
        source_major_online_attention_bf16_with_backend_f32(
            &query,
            query_rows,
            &key,
            &value,
            source_rows,
            heads,
            head_dim,
            8,
            128,
            false,
            crate::kernels::native::attention_backend(),
            &mut OnlineAttentionScratch {
                scores: &mut scores,
                maximum: &mut maximum,
                denominator: &mut denominator,
            },
            &mut actual,
        )
        .expect("valid BF16 source-major attention");
        assert_close(&actual, &expected);
    }

    #[test]
    fn tiled_online_attention_matches_uniform_and_peaked_scores() {
        for &(query, key) in &[
            (&[0.0, 0.0][..], &[0.0, 0.0, 0.0, 0.0][..]),
            (&[100.0, 0.0][..], &[1.0, 0.0, 0.0, 0.0][..]),
        ] {
            let value = [1.0, 2.0, 3.0, 4.0];
            let mut expected = [0.0; 2];
            let mut materialized = [0.0; 2];
            scaled_dot_product_attention_f32(
                query,
                1,
                key,
                &value,
                2,
                1,
                2,
                &mut materialized,
                &mut expected,
            )
            .expect("valid materialized attention");
            let mut actual = [0.0; 2];
            let mut scores = [0.0; 1];
            let mut maximum = [0.0; 1];
            let mut denominator = [0.0; 1];
            tiled_online_scaled_dot_product_attention_f32(
                query,
                1,
                key,
                &value,
                2,
                1,
                2,
                1,
                1,
                &mut OnlineAttentionScratch {
                    scores: &mut scores,
                    maximum: &mut maximum,
                    denominator: &mut denominator,
                },
                &mut actual,
            )
            .expect("valid tiled attention");
            assert_close(&actual, &expected);
        }
    }

    #[test]
    fn shape_errors_are_reported() {
        assert!(affine_f32(&[1.0], 1, 1, &[1.0], None, 1, &mut []).is_err());
        assert!(softmax_f32(&mut []).is_err());
        assert!(softmax_f32(&mut [f32::NAN]).is_err());
    }
}
