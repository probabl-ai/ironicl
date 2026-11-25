//! `POST /v1/predict`'s request: its JSON, its checks against the model, and
//! the worker's query it becomes.

use std::collections::HashMap;
use std::marker::PhantomData;

use garde::{Unvalidated, Valid, Validate};
use serde::{
    Deserialize, Deserializer,
    de::{self, DeserializeSeed, MapAccess, SeqAccess},
};

use super::{
    error::ApiError,
    worker::{Outputs, Query},
};
use crate::preprocess::{columns::encode_value, labels::Value};
use crate::{DEFAULT_ALPHAS, Labels, RegressionOutput, TaskKind};

/// A classifier's `output_type` names; a regressor's are `RegressionOutput`'s.
const CLASSIFIER_OUTPUTS: [&str; 2] = ["labels", "probabilities"];
/// The most quantile levels a request may ask for: as many as the model's raw
/// quantiles. Each level costs a value per row of the whole batch.
const MAX_ALPHAS: usize = 999;

/// What a request is checked against: the model's task and features, and
/// the request size it allows.
pub(super) struct Schema {
    pub task: TaskKind,
    pub features: usize,
    /// Each feature's categories, or `None` for a numeric feature.
    pub categories: Vec<Option<Labels>>,
    /// Each training column's position, by name, when the training data had
    /// a distinct name for each.
    pub columns: Option<HashMap<String, usize>>,
    pub max_request_rows: usize,
}

impl Schema {
    pub fn new(
        task: TaskKind,
        categories: Vec<Option<Labels>>,
        feature_names: Option<Vec<String>>,
        max_request_rows: usize,
    ) -> Self {
        let features = categories.len();
        let columns = feature_names
            .map(|names| {
                names
                    .into_iter()
                    .enumerate()
                    .map(|(i, name)| (name, i))
                    .collect::<HashMap<_, _>>()
            })
            .filter(|columns| columns.len() == features);
        Self {
            task,
            features,
            categories,
            columns,
            max_request_rows,
        }
    }
}

/// One value in a row: a number, a boolean, or text. `null`, which is
/// pandas' missing value, is `None` in its place.
#[derive(Clone, Debug, PartialEq)]
enum Cell {
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(String),
}

impl<'de> Deserialize<'de> for Cell {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl de::Visitor<'_> for Visitor {
            type Value = Cell;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("null, a number, a boolean, or a string")
            }
            fn visit_bool<E>(self, value: bool) -> Result<Cell, E> {
                Ok(Cell::Bool(value))
            }
            fn visit_i64<E>(self, value: i64) -> Result<Cell, E> {
                Ok(Cell::Int(value))
            }
            #[expect(
                clippy::cast_precision_loss,
                reason = "Only integers above i64::MAX are rounded"
            )]
            fn visit_u64<E>(self, value: u64) -> Result<Cell, E> {
                Ok(i64::try_from(value).map_or(Cell::Float(value as f64), Cell::Int))
            }
            fn visit_f64<E>(self, value: f64) -> Result<Cell, E> {
                Ok(Cell::Float(value))
            }
            fn visit_str<E>(self, value: &str) -> Result<Cell, E> {
                Ok(Cell::Text(value.to_owned()))
            }
            fn visit_string<E>(self, value: String) -> Result<Cell, E> {
                Ok(Cell::Text(value))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

impl Cell {
    fn value(&self) -> Value<'_> {
        match self {
            Self::Int(value) => Value::Int(*value),
            Self::Float(value) => Value::Float(*value),
            Self::Bool(value) => Value::Bool(*value),
            Self::Text(value) => Value::Text(value),
        }
    }
}

/// Either already encoded into flat `f64` rows during streaming JSON parsing,
/// or bounded raw cells waiting for a trailing `"columns"` key.
enum RequestData {
    Encoded(Rows),
    Raw(Vec<Vec<Option<Cell>>>),
}

impl RequestData {
    fn rows(&self) -> usize {
        match self {
            Self::Encoded(rows) => rows.count,
            Self::Raw(rows) => rows.len(),
        }
    }
}

/// `POST /v1/predict`'s body: a table as pandas'
/// `to_json(orient="split", index=False)` writes it, and options named after
/// `TabICL`'s predict arguments.
#[derive(Validate)]
#[garde(context(Schema))]
pub(super) struct PredictRequest {
    /// The column names, in the order of each row's values. Without them,
    /// the values are in the training data's order.
    #[garde(custom(match_the_model))]
    columns: Option<Vec<String>>,
    /// The rows: numbers for numeric features, and for categorical ones the
    /// type of the training categories. `null`, pandas' missing value, is
    /// replaced by its feature's training mean, or for a categorical feature
    /// encoded as a category training never saw.
    #[garde(custom(fit_the_model))]
    data: RequestData,
    #[garde(length(min = 1), custom(for_the_task_once))]
    output_type: Option<Vec<String>>,
    #[garde(custom(for_a_regressor), length(min = 1, max = MAX_ALPHAS), inner(inner(custom(between_0_and_1))))]
    alphas: Option<Vec<f32>>,
}

/// Deserialize an optional field using `S` when non-null.
struct OptionSeed<S>(S);

impl<'de, S: DeserializeSeed<'de>> DeserializeSeed<'de> for OptionSeed<S> {
    type Value = Option<S::Value>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        struct Visitor<S>(S);
        impl<'de, S: DeserializeSeed<'de>> de::Visitor<'de> for Visitor<S> {
            type Value = Option<S::Value>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("an optional value")
            }
            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(None)
            }
            fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
                self.0.deserialize(deserializer).map(Some)
            }
        }
        deserializer.deserialize_option(Visitor(self.0))
    }
}

/// Deserialize a `Vec<T>` stopping as soon as more than `max_len` elements appear.
struct BoundedVecSeed<T> {
    max_len: usize,
    over_limit: String,
    _marker: PhantomData<T>,
}

impl<T> BoundedVecSeed<T> {
    fn new(max_len: usize, over_limit: impl Into<String>) -> Self {
        Self {
            max_len,
            over_limit: over_limit.into(),
            _marker: PhantomData,
        }
    }
}

impl<'de, T: Deserialize<'de>> DeserializeSeed<'de> for BoundedVecSeed<T> {
    type Value = Vec<T>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        struct Visitor<T> {
            max_len: usize,
            over_limit: String,
            _marker: PhantomData<T>,
        }
        impl<'de, T: Deserialize<'de>> de::Visitor<'de> for Visitor<T> {
            type Value = Vec<T>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a sequence")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element::<T>()? {
                    if values.len() >= self.max_len {
                        return Err(de::Error::custom(&self.over_limit));
                    }
                    values.push(value);
                }
                Ok(values)
            }
        }
        deserializer.deserialize_seq(Visitor {
            max_len: self.max_len,
            over_limit: self.over_limit,
            _marker: PhantomData,
        })
    }
}

/// Stream-encode `"data"` directly into flat `f64` rows when the column mapping
/// is already known (e.g., `"columns"` preceded `"data"`, or the model has no
/// column names), enforcing row count and row width during deserialization.
struct EncodedDataSeed<'a> {
    schema: &'a Schema,
    /// Maps incoming column position `i` to the model's feature index.
    incoming_to_feature: Vec<usize>,
    columns: Option<&'a [String]>,
}

impl<'de> DeserializeSeed<'de> for EncodedDataSeed<'_> {
    type Value = Rows;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        struct RowSeed<'a> {
            schema: &'a Schema,
            incoming_to_feature: &'a [usize],
            columns: Option<&'a [String]>,
            row_index: usize,
            out: &'a mut [f64],
        }
        impl<'de> DeserializeSeed<'de> for RowSeed<'_> {
            type Value = ();
            fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
                struct Visitor<'a>(RowSeed<'a>);
                impl<'de> de::Visitor<'de> for Visitor<'_> {
                    type Value = ();
                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        formatter.write_str("a row of feature values")
                    }
                    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
                        let RowSeed {
                            schema,
                            incoming_to_feature,
                            columns,
                            row_index,
                            out,
                        } = self.0;
                        let mut count = 0_usize;
                        while let Some(cell) = seq.next_element::<Option<Cell>>()? {
                            if count >= schema.features {
                                return Err(de::Error::custom(format!(
                                    "data: row {row_index} has more than {} values; the model expects {}",
                                    schema.features, schema.features
                                )));
                            }
                            let feature = incoming_to_feature[count];
                            let value = encode_value(
                                cell.as_ref().map(Cell::value),
                                schema.categories[feature].as_ref(),
                            )
                            .map_err(|expected| {
                                let column = columns
                                    .map_or_else(|| count.to_string(), |names| format!("{:?}", names[count]));
                                de::Error::custom(format!(
                                    "data: row {row_index}, column {column} must be {expected} or null"
                                ))
                            })?;
                            out[feature] = value;
                            count += 1;
                        }
                        if count != schema.features {
                            return Err(de::Error::custom(format!(
                                "data: row {row_index} has {count} values; the model expects {}",
                                schema.features
                            )));
                        }
                        Ok(())
                    }
                }
                deserializer.deserialize_seq(Visitor(self))
            }
        }
        struct Visitor<'a>(EncodedDataSeed<'a>);
        impl<'de> de::Visitor<'de> for Visitor<'_> {
            type Value = Rows;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a sequence of rows")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Rows, A::Error> {
                let schema = self.0.schema;
                let mut values = Vec::new();
                let mut count = 0_usize;
                loop {
                    if count >= schema.max_request_rows {
                        if seq.next_element::<de::IgnoredAny>()?.is_some() {
                            return Err(de::Error::custom(format!(
                                "data: more than {} rows; the limit is {}",
                                schema.max_request_rows, schema.max_request_rows
                            )));
                        }
                        break;
                    }
                    let row_start = values.len();
                    values.resize(row_start + schema.features, 0.0);
                    let row = seq.next_element_seed(RowSeed {
                        schema,
                        incoming_to_feature: &self.0.incoming_to_feature,
                        columns: self.0.columns,
                        row_index: count,
                        out: &mut values[row_start..],
                    })?;
                    if row.is_none() {
                        values.truncate(row_start);
                        break;
                    }
                    count += 1;
                }
                Ok(Rows { values, count })
            }
        }
        deserializer.deserialize_seq(Visitor(self))
    }
}

/// Deserialize `"data"` into bounded raw cells when `"columns"` has not been
/// seen yet and the model supports named columns.
struct RawDataSeed<'a>(&'a Schema);

impl<'de> DeserializeSeed<'de> for RawDataSeed<'_> {
    type Value = Vec<Vec<Option<Cell>>>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        struct RawRowSeed {
            row_index: usize,
            features: usize,
        }
        impl<'de> DeserializeSeed<'de> for RawRowSeed {
            type Value = Vec<Option<Cell>>;
            fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
                struct Visitor(RawRowSeed);
                impl<'de> de::Visitor<'de> for Visitor {
                    type Value = Vec<Option<Cell>>;
                    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        formatter.write_str("a row of feature values")
                    }
                    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                        let mut row = Vec::with_capacity(self.0.features.min(64));
                        while let Some(cell) = seq.next_element::<Option<Cell>>()? {
                            if row.len() >= self.0.features {
                                return Err(de::Error::custom(format!(
                                    "data: row {} has more than {} values; the model expects {}",
                                    self.0.row_index, self.0.features, self.0.features
                                )));
                            }
                            row.push(cell);
                        }
                        if row.len() != self.0.features {
                            return Err(de::Error::custom(format!(
                                "data: row {} has {} values; the model expects {}",
                                self.0.row_index,
                                row.len(),
                                self.0.features
                            )));
                        }
                        Ok(row)
                    }
                }
                deserializer.deserialize_seq(Visitor(self))
            }
        }
        struct Visitor<'a>(&'a Schema);
        impl<'de> de::Visitor<'de> for Visitor<'_> {
            type Value = Vec<Vec<Option<Cell>>>;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a sequence of rows")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut rows = Vec::new();
                loop {
                    if rows.len() >= self.0.max_request_rows {
                        if seq.next_element::<de::IgnoredAny>()?.is_some() {
                            return Err(de::Error::custom(format!(
                                "data: more than {} rows; the limit is {}",
                                self.0.max_request_rows, self.0.max_request_rows
                            )));
                        }
                        break;
                    }
                    let Some(row) = seq.next_element_seed(RawRowSeed {
                        row_index: rows.len(),
                        features: self.0.features,
                    })?
                    else {
                        break;
                    };
                    rows.push(row);
                }
                Ok(rows)
            }
        }
        deserializer.deserialize_seq(Visitor(self.0))
    }
}

/// Deserialize a [`PredictRequest`] against `schema`, enforcing row, column,
/// output, and quantile-level bounds during parsing so malformed bodies cannot
/// amplify memory before validation.
struct PredictRequestSeed<'a>(&'a Schema);

impl<'de> DeserializeSeed<'de> for PredictRequestSeed<'_> {
    type Value = PredictRequest;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        const FIELDS: &[&str] = &["columns", "data", "output_type", "alphas"];
        struct Visitor<'a>(&'a Schema);
        impl<'de> de::Visitor<'de> for Visitor<'_> {
            type Value = PredictRequest;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a predict request object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<PredictRequest, A::Error> {
                let schema = self.0;
                let mut columns: Option<Option<Vec<String>>> = None;
                let mut data: Option<RequestData> = None;
                let mut output_type: Option<Option<Vec<String>>> = None;
                let mut alphas: Option<Option<Vec<f32>>> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "columns" => {
                            if columns.is_some() {
                                return Err(de::Error::duplicate_field("columns"));
                            }
                            let value = map.next_value_seed(OptionSeed(BoundedVecSeed::new(
                                schema.features + 1,
                                "columns: too many columns",
                            )))?;
                            columns = Some(value);
                        }
                        "data" => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            let parsed = match (&columns, &schema.columns) {
                                (Some(Some(names)), Some(positions)) => {
                                    match_the_model(&Some(names.clone()), schema)
                                        .map_err(|error| de::Error::custom(format!("columns: {error}")))?;
                                    let incoming_to_feature =
                                        names.iter().map(|name| positions[name]).collect();
                                    RequestData::Encoded(map.next_value_seed(EncodedDataSeed {
                                        schema,
                                        incoming_to_feature,
                                        columns: Some(names),
                                    })?)
                                }
                                (Some(Some(names)), None) => {
                                    if let Err(error) = match_the_model(&Some(names.clone()), schema) {
                                        return Err(de::Error::custom(format!("columns: {error}")));
                                    }
                                    RequestData::Encoded(map.next_value_seed(EncodedDataSeed {
                                        schema,
                                        incoming_to_feature: (0..schema.features).collect(),
                                        columns: None,
                                    })?)
                                }
                                (Some(None), _) | (None, None) => {
                                    RequestData::Encoded(map.next_value_seed(EncodedDataSeed {
                                        schema,
                                        incoming_to_feature: (0..schema.features).collect(),
                                        columns: None,
                                    })?)
                                }
                                (None, Some(_)) => {
                                    RequestData::Raw(map.next_value_seed(RawDataSeed(schema))?)
                                }
                            };
                            data = Some(parsed);
                        }
                        "output_type" => {
                            if output_type.is_some() {
                                return Err(de::Error::duplicate_field("output_type"));
                            }
                            let value = map.next_value_seed(OptionSeed(BoundedVecSeed::new(
                                8,
                                "output_type: too many outputs",
                            )))?;
                            output_type = Some(value);
                        }
                        "alphas" => {
                            if alphas.is_some() {
                                return Err(de::Error::duplicate_field("alphas"));
                            }
                            let value = map.next_value_seed(OptionSeed(BoundedVecSeed::new(
                                MAX_ALPHAS,
                                format!("alphas: length is greater than {MAX_ALPHAS}"),
                            )))?;
                            alphas = Some(value);
                        }
                        other => return Err(de::Error::unknown_field(other, FIELDS)),
                    }
                }
                Ok(PredictRequest {
                    columns: columns.flatten(),
                    data: data.ok_or_else(|| de::Error::missing_field("data"))?,
                    output_type: output_type.flatten(),
                    alphas: alphas.flatten(),
                })
            }
        }
        deserializer.deserialize_map(Visitor(self.0))
    }
}

#[expect(clippy::ref_option, reason = "garde passes the field as it is")]
fn match_the_model(columns: &Option<Vec<String>>, schema: &Schema) -> garde::Result {
    let Some(columns) = columns else { return Ok(()) };
    let Some(positions) = &schema.columns else {
        return Err(garde::Error::new(
            "the model was fit without column names; leave columns out",
        ));
    };
    let mut seen = vec![false; schema.features];
    for column in columns {
        let Some(&i) = positions.get(column) else {
            return Err(garde::Error::new(format!("unknown column {column:?}")));
        };
        if std::mem::replace(&mut seen[i], true) {
            return Err(garde::Error::new(format!("names column {column:?} twice")));
        }
    }
    if let Some((name, _)) = positions.iter().find(|&(_, &i)| !seen[i]) {
        return Err(garde::Error::new(format!("missing column {name:?}")));
    }
    Ok(())
}

fn fit_the_model(data: &RequestData, schema: &Schema) -> garde::Result {
    let count = data.rows();
    if count == 0 {
        return Err(garde::Error::new("has no rows"));
    }
    if count > schema.max_request_rows {
        return Err(garde::Error::new(format!(
            "{count} rows; the limit is {}",
            schema.max_request_rows
        )));
    }
    if let RequestData::Raw(rows) = data {
        for (i, row) in rows.iter().enumerate() {
            if row.len() != schema.features {
                return Err(garde::Error::new(format!(
                    "row {i} has {} values; the model expects {}",
                    row.len(),
                    schema.features
                )));
            }
        }
    }
    Ok(())
}

#[expect(clippy::ref_option, reason = "garde passes the field as it is")]
fn for_the_task_once(outputs: &Option<Vec<String>>, schema: &Schema) -> garde::Result {
    let Some(outputs) = outputs else { return Ok(()) };
    let known = |name: &str| match schema.task {
        TaskKind::Classifier => CLASSIFIER_OUTPUTS.contains(&name),
        TaskKind::Regressor => RegressionOutput::parse(name, &[0.5]).is_ok(),
    };
    if let Some(name) = outputs.iter().find(|name| !known(name)) {
        return Err(garde::Error::new(format!(
            "a {} has no output {name:?}",
            schema.task
        )));
    }
    if (1..outputs.len()).any(|i| outputs[..i].contains(&outputs[i])) {
        return Err(garde::Error::new("names an output twice"));
    }
    Ok(())
}

#[expect(clippy::ref_option, reason = "garde passes the field as it is")]
fn for_a_regressor(alphas: &Option<Vec<f32>>, schema: &Schema) -> garde::Result {
    match (alphas, schema.task) {
        (Some(_), TaskKind::Classifier) => Err(garde::Error::new("only a regressor takes alphas")),
        _ => Ok(()),
    }
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "garde passes each value by reference"
)]
fn between_0_and_1(alpha: &f32, _: &Schema) -> garde::Result {
    if *alpha > 0.0 && *alpha < 1.0 {
        Ok(())
    } else {
        Err(garde::Error::new(format!("must be between 0 and 1, got {alpha}")))
    }
}

impl PredictRequest {
    /// Parse and validate a JSON body against `schema` without materializing
    /// oversize arrays before checking limits.
    pub(super) fn parse(body: &[u8], schema: &Schema) -> Result<Query, ApiError> {
        let mut deserializer = serde_json::Deserializer::from_slice(body);
        let request = PredictRequestSeed(schema)
            .deserialize(&mut deserializer)
            .map_err(|error| ApiError::invalid(error.to_string()))?;
        deserializer
            .end()
            .map_err(|error| ApiError::invalid(error.to_string()))?;
        Self::into_query(request.check(schema)?, schema)
    }

    /// Check the request against `schema`.
    pub(super) fn check(self, schema: &Schema) -> Result<Valid<Self>, ApiError> {
        Unvalidated::new(self)
            .validate_with(schema)
            .map_err(|report| ApiError::invalid(report.to_string().trim_end()))
    }

    /// A checked request as the worker's query.
    ///
    /// # Errors
    ///
    /// When a value's type does not fit its feature.
    pub(super) fn into_query(request: Valid<Self>, schema: &Schema) -> Result<Query, ApiError> {
        let request = request.into_inner();
        let rows = match request.data {
            RequestData::Encoded(rows) => rows,
            RequestData::Raw(raw) => {
                // Where each of the model's features is in a row.
                let mut order: Vec<usize> = (0..schema.features).collect();
                if let (Some(columns), Some(positions)) = (&request.columns, &schema.columns) {
                    for (i, column) in columns.iter().enumerate() {
                        order[positions[column]] = i;
                    }
                }
                let mut values = Vec::with_capacity(raw.len() * schema.features);
                for (r, row) in raw.iter().enumerate() {
                    for (&i, categories) in order.iter().zip(&schema.categories) {
                        let value = encode_value(row[i].as_ref().map(Cell::value), categories.as_ref())
                            .map_err(|expected| {
                                let column = request
                                    .columns
                                    .as_ref()
                                    .map_or_else(|| i.to_string(), |names| format!("{:?}", names[i]));
                                ApiError::invalid(format!(
                                    "data: row {r}, column {column} must be {expected} or null"
                                ))
                            })?;
                        values.push(value);
                    }
                }
                Rows {
                    values,
                    count: raw.len(),
                }
            }
        };
        let outputs = match schema.task {
            TaskKind::Classifier => {
                // Without output_type, both.
                let wants = |output: &str| {
                    (request.output_type.as_ref()).is_none_or(|names| names.iter().any(|name| name == output))
                };
                Outputs::Classifier {
                    labels: wants("labels"),
                    probabilities: wants("probabilities"),
                }
            }
            TaskKind::Regressor => {
                let alphas = request.alphas.unwrap_or_else(|| DEFAULT_ALPHAS.to_vec());
                let outputs = match request.output_type {
                    Some(names) => names
                        .iter()
                        .map(|name| RegressionOutput::parse(name, &alphas))
                        .collect::<crate::Result<_>>()?,
                    None => vec![RegressionOutput::Mean],
                };
                Outputs::Regressor { outputs, alphas }
            }
        };
        Ok(Query { rows, outputs })
    }
}

/// `count` rows of missing values with the default outputs, to warm a `task`
/// model whose features have `categories` up with.
pub(super) fn missing_rows(
    task: TaskKind,
    categories: Vec<Option<Labels>>,
    count: usize,
) -> Result<Query, ApiError> {
    let features = categories.len();
    let schema = Schema::new(task, categories, None, count);
    let request = PredictRequest {
        columns: None,
        data: RequestData::Raw(vec![vec![None; features]; count]),
        output_type: None,
        alphas: None,
    };
    PredictRequest::into_query(request.check(&schema)?, &schema)
}

/// A query's rows, flattened in the training data's column order.
pub(super) struct Rows {
    pub values: Vec<f64>,
    pub count: usize,
}

#[cfg(test)]
mod tests {
    use super::{MAX_ALPHAS, PredictRequest, Schema};
    use crate::{Labels, TaskKind};

    /// The schema of a `task` model with the numeric columns `a` and `b`.
    fn schema(task: TaskKind) -> Schema {
        Schema::new(task, vec![None; 2], Some(vec!["a".into(), "b".into()]), 3)
    }

    /// The error that parsing, checking, and reading `json` against `schema`
    /// gives, or `""`.
    fn error(schema: &Schema, json: &str) -> String {
        PredictRequest::parse(json.as_bytes(), schema)
            .err()
            .map_or_else(String::new, |error| error.message().to_owned())
    }

    /// The values that `json`'s rows become.
    fn values(schema: &Schema, json: &str) -> Vec<f64> {
        PredictRequest::parse(json.as_bytes(), schema)
            .unwrap()
            .rows
            .values
    }

    #[test]
    fn categorical_values_become_category_indices() {
        let categories = vec![
            Some(Labels::Text(vec!["x".into(), "y".into()])),
            None,
            Some(Labels::Int(vec![2, 10])),
        ];
        let names = ["c", "n", "i"].map(String::from).to_vec();
        let schema = Schema::new(TaskKind::Classifier, categories, Some(names), 3);
        let got = values(
            &schema,
            r#"{"data": [["y", 1.5, 10], ["z", null, 2.0], [null, 3, 7]]}"#,
        );
        let expected = [1.0, 1.5, 1.0, -1.0, f64::NAN, 0.0, -1.0, 3.0, -1.0];
        assert!(
            got.iter().zip(expected).all(|(a, b)| a.total_cmp(&b).is_eq()),
            "{got:?}"
        );
        assert_eq!(
            values(&schema, r#"{"columns": ["n", "i", "c"], "data": [[1, 2, "x"]]}"#),
            [0.0, 1.0, 0.0]
        );
        for (json, message) in [
            (
                r#"{"data": [[1, 1, 2]]}"#,
                "row 0, column 0 must be a string or null",
            ),
            (
                r#"{"data": [["x", "1", 2]]}"#,
                "row 0, column 1 must be a number or null",
            ),
            (
                r#"{"columns": ["n", "i", "c"], "data": [[1, true, "x"]]}"#,
                r#"row 0, column "i" must be a number or null"#,
            ),
        ] {
            assert!(
                error(&schema, json).contains(message),
                "{json}: {}",
                error(&schema, json)
            );
        }
    }

    #[test]
    fn columns_put_the_values_in_the_training_order() {
        let schema = schema(TaskKind::Classifier);
        for (json, values) in [
            (r#"{"data": [[1, 2], [3, 4]]}"#, [1.0, 2.0, 3.0, 4.0]),
            (
                r#"{"columns": ["a", "b"], "data": [[1, 2], [3, 4]]}"#,
                [1.0, 2.0, 3.0, 4.0],
            ),
            (
                r#"{"columns": ["b", "a"], "data": [[1, 2], [3, 4]]}"#,
                [2.0, 1.0, 4.0, 3.0],
            ),
        ] {
            assert_eq!(super::tests::values(&schema, json), values.to_vec(), "{json}");
        }
    }

    #[test]
    fn requests_are_checked_against_the_model() {
        use TaskKind::{Classifier, Regressor};
        let alphas = vec!["0.5"; MAX_ALPHAS + 1].join(", ");
        let too_many_alphas = format!(r#"{{"data": [[1, 2]], "alphas": [{alphas}]}}"#);
        for (task, json, message) in [
            (Classifier, r#"{"data": [[1, 2]], "output_type": ["labels"]}"#, ""),
            (
                Regressor,
                r#"{"data": [[1, 2]], "output_type": ["quantiles"], "alphas": [0.5]}"#,
                "",
            ),
            (Classifier, r#"{"data": []}"#, "data: has no rows"),
            (
                Classifier,
                r#"{"data": [[1, 2], [3]]}"#,
                "data: row 1 has 1 values; the model expects 2",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2], [1, 2], [1, 2], [1, 2]]}"#,
                "data: more than 3 rows; the limit is 3",
            ),
            (Classifier, r#"{"data": [[1, null]]}"#, ""),
            (
                Classifier,
                r#"{"data": [[1, "x"]]}"#,
                "row 0, column 1 must be a number or null",
            ),
            (
                Classifier,
                r#"{"data": [[1, [2]]]}"#,
                "expected null, a number, a boolean, or a string",
            ),
            (Classifier, r#"{"data": [1]}"#, "invalid type: integer"),
            (Classifier, r#"{"rows": [[1, 2]]}"#, "unknown field `rows`"),
            (
                Classifier,
                r#"{"columns": ["a", "c"], "data": [[1, 2]]}"#,
                "columns: unknown column \"c\"",
            ),
            (
                Classifier,
                r#"{"columns": ["a"], "data": [[1, 2]]}"#,
                "columns: missing column \"b\"",
            ),
            (
                Classifier,
                r#"{"columns": ["a", "b", "a"], "data": [[1, 2]]}"#,
                "columns: names column \"a\" twice",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2]], "output_type": ["mean"]}"#,
                "a classifier has no output \"mean\"",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2]], "output_type": ["average"]}"#,
                "a classifier has no output \"average\"",
            ),
            (
                Regressor,
                r#"{"data": [[1, 2]], "output_type": ["labels"]}"#,
                "a regressor has no output \"labels\"",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2]], "output_type": []}"#,
                "output_type: length is lower than 1",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2]], "output_type": ["labels", "labels"]}"#,
                "names an output twice",
            ),
            (
                Classifier,
                r#"{"data": [[1, 2]], "alphas": [0.5]}"#,
                "only a regressor takes alphas",
            ),
            (
                Regressor,
                r#"{"data": [[1, 2]], "alphas": [0.5, 1]}"#,
                "between 0 and 1, got 1",
            ),
            (Regressor, &too_many_alphas, "alphas: length is greater than 999"),
        ] {
            let got = error(&schema(task), json);
            if message.is_empty() {
                assert_eq!(got, "", "{json}");
            } else {
                assert!(got.contains(message), "{json}: {got}");
            }
        }
        // Names that are not one per column are no names at all.
        let repeated = Schema::new(Classifier, vec![None; 2], Some(vec!["a".into(), "a".into()]), 3);
        assert!(repeated.columns.is_none());
        let unnamed = Schema::new(Classifier, vec![None; 2], None, 3);
        assert!(
            error(&unnamed, r#"{"columns": ["a", "b"], "data": [[1, 2]]}"#)
                .contains("fit without column names")
        );
    }

    #[test]
    fn oversize_arrays_and_trailing_columns_are_handled_boundedly() {
        let s = schema(TaskKind::Classifier);
        assert!(
            error(&s, r#"{"data": [[1, 2, 3]]}"#)
                .contains("data: row 0 has more than 2 values; the model expects 2")
        );
        assert!(
            error(&s, r#"{"columns": ["a", "b", "c", "d"], "data": [[1, 2]]}"#)
                .contains("columns: too many columns")
        );
        assert_eq!(
            values(&s, r#"{"data": [[1, 2], [3, 4]], "columns": ["b", "a"]}"#),
            vec![2.0, 1.0, 4.0, 3.0]
        );
    }
}
