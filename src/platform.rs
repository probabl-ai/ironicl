use std::fmt;

/// Runtime CPU capabilities relevant to Ironicl kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct CpuCapabilities {
    pub architecture: &'static str,
    pub operating_system: &'static str,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512bf16: bool,
    pub avx512vnni: bool,
}

impl CpuCapabilities {
    #[must_use]
    pub fn detect() -> Self {
        Self::detect_for_target()
    }

    #[cfg(target_arch = "x86_64")]
    fn detect_for_target() -> Self {
        Self {
            architecture: std::env::consts::ARCH,
            operating_system: std::env::consts::OS,
            avx2: std::is_x86_feature_detected!("avx2"),
            avx512f: std::is_x86_feature_detected!("avx512f"),
            avx512bw: std::is_x86_feature_detected!("avx512bw"),
            avx512vl: std::is_x86_feature_detected!("avx512vl"),
            avx512bf16: std::is_x86_feature_detected!("avx512bf16"),
            avx512vnni: std::is_x86_feature_detected!("avx512vnni"),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn detect_for_target() -> Self {
        Self {
            architecture: std::env::consts::ARCH,
            operating_system: std::env::consts::OS,
            avx2: false,
            avx512f: false,
            avx512bw: false,
            avx512vl: false,
            avx512bf16: false,
            avx512vnni: false,
        }
    }

    #[must_use]
    pub const fn supports_zen5_base(&self) -> bool {
        self.avx2 && self.avx512f && self.avx512bw && self.avx512vl
    }
}

impl fmt::Display for CpuCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "architecture:    {}", self.architecture)?;
        writeln!(f, "operating system: {}", self.operating_system)?;
        writeln!(f, "avx2:            {}", yes_no(self.avx2))?;
        writeln!(f, "avx512f:         {}", yes_no(self.avx512f))?;
        writeln!(f, "avx512bw:        {}", yes_no(self.avx512bw))?;
        writeln!(f, "avx512vl:        {}", yes_no(self.avx512vl))?;
        writeln!(f, "avx512bf16:      {}", yes_no(self.avx512bf16))?;
        write!(f, "avx512vnni:      {}", yes_no(self.avx512vnni))
    }
}

const fn yes_no(value: bool) -> &'static str {
    if value { "yes" } else { "no" }
}
