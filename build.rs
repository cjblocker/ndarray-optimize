use rustc_version::{version, version_meta, Channel};

fn main() {
    let ver = version().unwrap();
    assert!(ver.major >= 1);

    match version_meta().unwrap().channel {
        Channel::Nightly => {
            println!("cargo:rustc-cfg=rustc_nightly");
        }
        Channel::Beta => {
            println!("cargo:rustc-cfg=rustc_beta");
        }
        _ => {}
    }
}
