use rustc_version::{version_meta, Channel};

fn main() {
    println!("cargo:rustc-check-cfg=cfg(knyst_nightly)");
    let channel = version_meta().map(|meta| meta.channel);
    if matches!(channel, Ok(Channel::Nightly) | Ok(Channel::Dev)) {
        println!("cargo:rustc-cfg=knyst_nightly");
    }
}
