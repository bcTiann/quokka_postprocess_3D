module QuokkaProcess

using JSON3
using Mera
using NPZ
using SHA
using SpecialFunctions

include("reader.jl")
include("emission.jl")
include("products.jl")
include("stream_reader.jl")
include("stream_products.jl")
include("pipeline.jl")

const MERA_REVISION = "db277419e05a45e8d4921fb5f3a0a4d15792cf7f"

"""Record source identities without hashing multi-gigabyte snapshot payloads."""
function source_identity(dataset, assets)
    snapshot_files = Dict{String,Any}()
    for (directory, _, files) in walkdir(dataset; follow_symlinks=true)
        for name in files
            path = joinpath(directory, name)
            state = stat(path)
            snapshot_files[relpath(path, dataset)] = (size=state.size, mtime=state.mtime)
        end
    end
    sources = Dict(
        name => bytes2hex(sha256(read(joinpath(@__DIR__, name))))
        for name in readdir(@__DIR__) if endswith(name, ".jl")
    )
    project = dirname(@__DIR__)
    python_sources = Dict{String,String}()
    for path in (joinpath(project, "prepare_results.py"), joinpath(project, "plot_results.py"))
        python_sources[relpath(path, project)] = bytes2hex(sha256(read(path)))
    end
    package_root = joinpath(dirname(project), "src", "quokka2s")
    for (directory, _, files) in walkdir(package_root)
        for name in files
            (endswith(name, ".py") || endswith(name, ".pyx")) || continue
            path = joinpath(directory, name)
            python_sources["quokka2s/" * relpath(path, package_root)] = bytes2hex(sha256(read(path)))
        end
    end
    return Dict(
        "snapshot_file_metadata" => snapshot_files,
        "julia_source_sha256" => sources,
        "python_preparation_and_renderer_source_sha256" => python_sources,
        "physics_json_sha256" => bytes2hex(sha256(read(joinpath(assets, "physics.json")))),
        "tables_npz_sha256" => open(source -> bytes2hex(sha256(source)), joinpath(assets, "tables.npz")),
        "project_sha256" => bytes2hex(sha256(read(joinpath(project, "Project.toml")))),
        "manifest_sha256" => bytes2hex(sha256(read(joinpath(project, "Manifest.toml")))),
        "mera_revision" => MERA_REVISION,
        "julia_version" => string(VERSION),
    )
end

function write_json(path, record)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        JSON3.pretty(io, record)
        println(io)
    end
    mv(temporary, path; force=true)
end

"""Save rendering coordinates, limits and normalized curves before plotting."""
function prepare_display!(products, tile, meta)
    parsec_cm = Float64(meta["constants"]["parsec_cm"])
    for (axis, range, width, origin) in (
        ("x", tile.x_range, tile.cell_width_cm[1], tile.domain_left_cm[1]),
        ("y", tile.y_range, tile.cell_width_cm[2], tile.domain_left_cm[2]),
    )
        indices = (first(range) - 1):last(range)
        products[axis * "_edges_kpc"] = (origin .+ collect(indices) .* width) ./ (1000 * parsec_cm)
    end
    surface_cm2 = products["spectra_total_surface_luminosity_erg_s_cm2_per_kms"]
    products["spectra_surface_erg_s_kpc2_per_kms"] = surface_cm2 .* (1000 * parsec_cm)^2
    images = products["images_luminosity_erg_s"]
    display = copy(images)
    limits = zeros(size(images, 3), size(images, 4), 2)
    for dust in axes(images, 4), line in axes(images, 3)
        limits[line, dust, :] .= products["image_color_limits_erg_s"][line, :]
        # A physically empty line still needs a valid logarithmic display scale.
        if limits[line, dust, 2] <= 0
            limits[line, dust, :] .= (1.0, 10.0)
        end
        panel = @view display[:, :, line, dust]
        panel[panel .<= 0] .= NaN
    end
    products["image_display_luminosity_erg_s"] = display
    products["image_display_limits_erg_s"] = limits
    return products
end

"""Check independent image, channel and cell luminosities for every line/state."""
function check_products(products, emission, tile, meta)
    line_count = size(emission.emissivity_erg_s_cm3, 2)
    expected = zeros(line_count, 2, 2)
    foreground = vec(tile.foreground_NH_cm2)
    opacity = Vector{Float64}(meta["dust_cross_sections_cm2_H"])
    for line in 1:line_count, cell in axes(emission.emissivity_erg_s_cm3, 1)
        epsilon = emission.emissivity_erg_s_cm3[cell, line]
        isfinite(epsilon) || continue
        epsilon >= 0 || error("Negative cell emissivity")
        branch = emission.cold_cells[cell] ? 1 : 2
        luminosity = epsilon * tile.cell_volume_cm3
        expected[line, 1, branch] += luminosity
        expected[line, 2, branch] += luminosity * exp(-opacity[line] * foreground[cell])
    end
    function compare(name, actual, reference)
        size(actual) == size(reference) || error("Shape mismatch in $name")
        all(isfinite, actual) || error("Nonfinite values in $name")
        residual = 0.0
        for index in eachindex(reference)
            target = reference[index]
            difference = abs(actual[index] - target)
            target == 0 ? (difference == 0 || error("Nonzero physical zero in $name")) :
                (difference <= 1e-10 * abs(target) || error("Conservation failure in $name"))
            target != 0 && (residual = max(residual, difference / abs(target)))
        end
        return residual
    end
    total_expected = dropdims(sum(expected; dims=3); dims=3)
    residuals = Dict(
        "cell_luminosity" => compare("cell luminosity", products["input_luminosity_erg_s"], expected),
        "image_luminosity" => compare("image luminosity", products["image_total_luminosity_erg_s"], total_expected),
        "window_plus_outside" => compare("window plus outside", products["captured_luminosity_erg_s"] + products["outside_velocity_luminosity_erg_s"], expected),
        "full_moment_luminosity" => compare("full moments", products["full_line_luminosity_erg_s"], total_expected),
    )
    counts = products["phase_cell_count"]
    sum(counts[1:end-1]) == counts[end] || error("Phase cell accounting failed")
    counts[end] + only(products["phase_missing_temperature_count"]) == length(foreground) || error("Gas cell accounting failed")
    expected_mass = sum(tile.density_g_cm3) * tile.cell_volume_cm3
    recovered_mass = last(products["phase_mass_g"]) + only(products["phase_missing_temperature_mass_g"])
    residuals["gas_mass"] = compare("independent gas mass", [recovered_mass], [expected_mass])
    return residuals
end

"""Resolve existing ancestors so output cannot be created inside either input."""
function output_destination(output, input_directories)
    output = abspath(output)
    ispath(output) && error("Output already exists and was preserved: $output")
    ancestor = dirname(output)
    suffix = [basename(output)]
    while !ispath(ancestor)
        pushfirst!(suffix, basename(ancestor))
        ancestor = dirname(ancestor)
    end
    output = joinpath(realpath(ancestor), suffix...)
    for input in input_directories
        relative = relpath(output, input)
        inside = relative == "." || first(splitpath(relative)) != ".."
        inside && error("Output must be outside input directory: $input")
    end
    return output
end

"""Validate an explicitly supplied simulation label without inferring parent folders."""
function checked_jobname(jobname)
    jobname isa AbstractString || error("jobname must be a nonempty string")
    !isempty(strip(jobname)) || error("jobname must be a nonempty string")
    jobname in (".", "..") && error("jobname must name a simulation, not a path")
    any(character -> character in ('/', '\\'), jobname) && error("jobname must not contain path separators")
    return String(jobname)
end

"""
Process one explicit complete-z region using native Julia numerical stages.

Ranges use Julia's inclusive 1-based native indices. The command-line interface
accepts 0-based half-open indices instead. This initial implementation retains
one region in memory; max_cells rejects oversized requests before field I/O.
"""
function run_region(; dataset, assets, output, x_range, y_range, jobname=nothing,
                    threads=Threads.nthreads(), max_cells=2_000_000, save_cells=false)
    jobname !== nothing && (jobname = checked_jobname(jobname))
    dataset = realpath(dataset)
    assets = realpath(assets)
    output = output_destination(output, (dataset, assets))
    1 <= threads <= Threads.nthreads() || error("Requested threads exceed the Julia process thread count")
    began = time()
    println("Opening Mera metadata and fixed emission assets")
    snapshot = open_snapshot(dataset)
    cells = length(x_range) * length(y_range) * snapshot.original_shape[3]
    cells <= max_cells || error("Region has $cells cells; prototype limit is $max_cells")
    model = load_emission_model(assets)
    identity = source_identity(dataset, assets)
    mkpath(output)
    timing = Dict{String,Float64}("initialization_seconds" => time() - began)
    write_json(joinpath(output, "status.json"), Dict("status" => "running", "processing_complete" => false))
    try
        println("Reading complete-z region: $cells cells")
        stage = time()
        tile = read_tile(snapshot; x_range=x_range, y_range=y_range, physics=model.meta["constants"])
        timing["read_and_derived_fields_seconds"] = time() - stage
        println("Calculating ten-line emission and table diagnostics")
        stage = time()
        emission = calculate_emission(model, tile)
        timing["emission_seconds"] = time() - stage
        println("Accumulating images, Gaussian spectra and gas phases")
        stage = time()
        products = accumulate_products(tile, emission, model.meta; threads=threads)
        timing["products_seconds"] = time() - stage
        stage = time()
        residuals = check_products(products, emission, tile, model.meta)
        prepare_display!(products, tile, model.meta)
        timing["checks_and_preparation_seconds"] = time() - stage
        source_identity(dataset, assets) == identity || error("Inputs or implementation changed during processing")
        stage = time()
        NPZ.npzwrite(joinpath(output, "products.tmp.npz"), products)
        mv(joinpath(output, "products.tmp.npz"), joinpath(output, "products.npz"))
        if save_cells
            fields = Dict{String,Any}(String(name) => getproperty(tile, name) for name in (
                :nH_cm3, :temperature_quokka_K, :shielding_NH_cm2, :foreground_NH_cm2,
                :dVdr_s, :velocity_z_kms, :density_g_cm3,
            ))
            shape = size(tile.nH_cm3)
            fields["emissivity_erg_s_cm3"] = reshape(emission.emissivity_erg_s_cm3, shape..., :)
            fields["temperature_K"] = reshape(emission.temperature_K, shape..., :)
            fields["despotic_temperature_K"] = reshape(emission.despotic_temperature_K, shape)
            fields["cold_cells"] = reshape(emission.cold_cells, shape)
            NPZ.npzwrite(joinpath(output, "cells.npz"), fields)
        end
        timing["save_arrays_seconds"] = time() - stage
        timing["total_seconds"] = time() - began
        report = Dict(
            "status" => "completed", "processing_complete" => true,
            "implementation" => "Julia/Mera region prototype", "schema_version" => 1,
            "full_snapshot" => cells == prod(snapshot.original_shape),
            "dataset" => dataset, "jobname" => jobname, "snapshot_name" => basename(dataset),
            "time_s" => snapshot.time_s, "physics" => model.meta, "identity" => identity,
            "x_index_range" => [first(x_range) - 1, last(x_range)],
            "y_index_range" => [first(y_range) - 1, last(y_range)],
            "shape" => collect(size(tile.nH_cm3)), "cell_count" => cells,
            "cell_width_cm" => collect(tile.cell_width_cm),
            "domain_left_cm" => collect(tile.domain_left_cm),
            "reader_stats" => tile.reader_stats, "timing" => timing,
            "threads" => threads, "peak_rss_bytes" => Sys.maxrss(),
            "conservation_relative_residuals" => residuals,
            "emission_diagnostics" => emission.diagnostics,
            "axis_order" => Dict("images" => ["x", "y", "line", "dust"], "spectra" => ["channel", "line", "dust", "regime"]),
            "dust_states" => ["intrinsic", "attenuated"], "regimes" => ["cold", "hot"],
        )
        write_json(joinpath(output, "report.json"), report)
        write_json(joinpath(output, "status.json"), Dict("status" => "completed", "processing_complete" => true, "elapsed_seconds" => time() - began))
        println("Completed region: ", output, " (", round(time() - began; digits=3), " s)")
        return report
    catch exception
        write_json(joinpath(output, "status.json"), Dict("status" => "failed", "processing_complete" => false, "error" => sprint(showerror, exception), "elapsed_seconds" => time() - began))
        rethrow()
    end
end

export run_region, run_snapshot

end
