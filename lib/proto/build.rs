fn main() -> Result<(), Box<dyn std::error::Error>> {
    // build the request plane proto
    tonic_prost_build::configure()
        .compile_well_known_types(true)
        .extern_path(".google.protobuf", "::prost_wkt_types")
        .compile_protos(
            &[
                "protos/request_plane/v1/request_plane.proto",
            ],
            &["."],
        )?;

    Ok(())
}