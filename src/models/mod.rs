//! The models Ironicl runs, each with a graph per device.

pub mod checkpoint;
pub mod checkpoint_cache;
pub mod directory;
pub mod kumo_tabular;
pub mod quantiles;
pub mod tabicl_v2;

/// Which model owns the artifact state, independent of task, release or device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum ModelFamily {
    #[serde(rename = "tabicl_v2")]
    TabIclV2,
    #[serde(rename = "kumo_tabular")]
    KumoTabular,
}
