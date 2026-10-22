"""Add independently calculated cell luminosities to a small verification sum."""
function add_expected_luminosity!(expected, emission, tile, opacity::Vector{Float64})
    partial = zeros(size(expected))
    foreground = vec(tile.foreground_NH_cm2)
    values = emission.emissivity_erg_s_cm3
    volume = tile.cell_volume_cm3
    for line in axes(values, 2)
        extinction = opacity[line]
        for cell in axes(values, 1)
            epsilon = values[cell, line]
            isnan(epsilon) && continue
            isfinite(epsilon) && epsilon >= 0 || error("Invalid cell emissivity in independent accounting")
            branch = emission.cold_cells[cell] ? 1 : 2
            luminosity = epsilon * volume
            partial[line, 1, branch] += luminosity
            partial[line, 2, branch] += luminosity * exp(-extinction * foreground[cell])
        end
    end
    expected .+= partial
    return nothing
end

"""Accumulate integer interpolation diagnostics without retaining cell arrays."""
function merge_emission_diagnostics!(total, incoming)
    for (key, value) in incoming
        if value isa AbstractDict
            destination = get!(total, key, Dict{String,Any}())
            merge_emission_diagnostics!(destination, value)
        elseif value isa Integer
            total[key] = get(total, key, 0) + value
        else
            error("Unsupported emission diagnostic: $key")
        end
    end
    return total
end

"""Compare final products to separate cell sums, including zero and missing mass."""
function check_snapshot_products(products, expected, expected_mass, cell_count)
    function relative_error(name, actual, reference)
        size(actual) == size(reference) || error("Shape mismatch in $name")
        all(isfinite, actual) || error("Nonfinite values in $name")
        residual = 0.0
        for index in eachindex(reference)
            target = reference[index]
            difference = abs(actual[index] - target)
            if target == 0
                difference == 0 || error("Nonzero physical zero in $name")
            else
                difference <= 1e-10 * abs(target) || error("Conservation failure in $name")
                residual = max(residual, difference / abs(target))
            end
        end
        return residual
    end
    total_expected = dropdims(sum(expected; dims=3); dims=3)
    mass = last(products["phase_mass_g"]) + only(products["phase_missing_temperature_mass_g"])
    counts = products["phase_cell_count"]
    sum(counts[1:end-1]) == counts[end] || error("Phase cell accounting failed")
    counts[end] + only(products["phase_missing_temperature_count"]) == cell_count || error("Snapshot cell accounting failed")
    return Dict(
        "cell_luminosity" => relative_error("cell luminosity", products["input_luminosity_erg_s"], expected),
        "image_luminosity" => relative_error("image luminosity", products["image_total_luminosity_erg_s"], total_expected),
        "window_plus_outside" => relative_error("window plus outside", products["captured_luminosity_erg_s"] + products["outside_velocity_luminosity_erg_s"], expected),
        "full_moment_luminosity" => relative_error("full moments", products["full_line_luminosity_erg_s"], total_expected),
        "gas_mass" => relative_error("independent gas mass", [mass], [expected_mass]),
    )
end

function default_python()
    local_python = normpath(joinpath(@__DIR__, "..", "..", ".venv", "bin", "python"))
    isfile(local_python) && return local_python
    executable = Sys.which("python3")
    executable === nothing && error("Python is required for saved-product preparation and plotting; use --python PATH")
    return executable
end

"""Run the saved-array preparation or renderer without a shell or input re-reading."""
function run_python_stage(python, script, arguments)
    script_path = normpath(joinpath(@__DIR__, "..", script))
    run(Cmd(vcat([String(python), script_path], String.(arguments))))
    return nothing
end

"""Write optional per-batch verification fields; never retain them across batches."""
function save_batch_cells(output, batch_number, tile, emission)
    directory = joinpath(output, "cells")
    mkpath(directory)
    arrays = Dict{String,Any}(String(name) => getproperty(tile, name) for name in (
        :nH_cm3, :temperature_quokka_K, :shielding_NH_cm2, :foreground_NH_cm2,
        :dVdr_s, :velocity_z_kms, :density_g_cm3,
    ))
    shape = size(tile.nH_cm3)
    arrays["emissivity_erg_s_cm3"] = reshape(emission.emissivity_erg_s_cm3, shape..., :)
    arrays["temperature_K"] = reshape(emission.temperature_K, shape..., :)
    arrays["despotic_temperature_K"] = reshape(emission.despotic_temperature_K, shape)
    arrays["cold_cells"] = reshape(emission.cold_cells, shape)
    arrays["z_index_range"] = [first(tile.z_range) - 1, last(tile.z_range)]
    name = "batch_" * lpad(string(batch_number), 6, '0') * ".npz"
    NPZ.npzwrite(joinpath(directory, name), arrays)
    return nothing
end

"""
Process a complete uniform snapshot, or an explicit XY region, with bounded batches.

The reader streams original z blocks and supplies complete-column shielding and
foreground values. Only products and small diagnostics survive each callback.
Figure preparation is part of numerical processing; rendering is optional.
"""
function run_snapshot(; dataset, jobname, assets, output, x_range=nothing, y_range=nothing,
                      threads=Threads.nthreads(), batch_cells=131_072,
                      reader_cache_bytes=1024^3, python=default_python(),
                      make_plots=true, figure_formats=["png"],
                      figure_styles=["titled"], image_downsample_factor=1,
                      raw_luminosity=false, save_cells=false)
    jobname = checked_jobname(jobname)
    dataset = realpath(dataset)
    assets = realpath(assets)
    output = output_destination(output, (dataset, assets))
    1 <= threads <= Threads.nthreads() || error("Requested threads exceed the Julia process thread count")
    batch_cells > 0 || error("batch_cells must be positive")
    reader_cache_bytes > 0 || error("reader_cache_bytes must be positive")
    image_downsample_factor > 0 || error("image_downsample_factor must be positive")
    all(format -> format in ("png", "pdf"), figure_formats) || error("Unsupported figure format")
    all(style -> style in ("titled", "untitled"), figure_styles) || error("Unsupported figure style")
    began = time()
    println("Opening Mera snapshot metadata and fixed emission assets")
    snapshot = open_snapshot(dataset)
    selected_x = x_range === nothing ? (1:snapshot.original_shape[1]) : x_range
    selected_y = y_range === nothing ? (1:snapshot.original_shape[2]) : y_range
    for (name, selected, count) in (("x", selected_x, snapshot.original_shape[1]), ("y", selected_y, snapshot.original_shape[2]))
        !isempty(selected) && 1 <= first(selected) <= last(selected) <= count || error("Invalid $name selection")
    end
    shape = (length(selected_x), length(selected_y), snapshot.original_shape[3])
    all(dimension -> dimension % image_downsample_factor == 0, shape[1:2]) || error("Image factor must divide both selected image dimensions")
    geometry = (
        x_range=selected_x,
        y_range=selected_y,
        z_range=1:snapshot.original_shape[3],
        cell_width_cm=snapshot.cell_width_cm,
        cell_volume_cm3=snapshot.cell_volume_cm3,
        domain_left_cm=snapshot.domain_left_cm,
    )
    model = load_emission_model(assets)
    opacity = Vector{Float64}(model.meta["dust_cross_sections_cm2_H"])
    accumulator = ProductAccumulator(geometry, model.meta; threads=threads)
    identity = source_identity(dataset, assets)
    run(Cmd([String(python), "-c", "import numpy; import matplotlib; import quokka2s.emission_results"]))
    mkpath(output)
    expected = zeros(length(opacity), 2, 2)
    expected_mass = 0.0
    diagnostics = Dict{String,Any}()
    timings = Dict{String,Float64}(
        "initialization_seconds" => time() - began,
        "emission_seconds" => 0.0,
        "products_seconds" => 0.0,
        "independent_accounting_seconds" => 0.0,
        "debug_save_seconds" => 0.0,
    )
    status = Dict{String,Any}(
        "status" => "running", "processing_complete" => false,
        "full_snapshot" => shape == snapshot.original_shape,
        "jobname" => jobname, "snapshot_name" => basename(dataset),
        "stage" => "density_prepass", "completed_cells" => 0,
        "total_cells" => prod(shape), "completed_batches" => 0,
    )
    last_print = Ref(0.0)
    last_stage = Ref("")
    main_started = Ref(0.0)
    function progress(stage; force=false, details=nothing)
        now = time()
        status["stage"] = stage
        status["elapsed_seconds"] = now - began
        details !== nothing && (status["reader_progress"] = details)
        completed = status["completed_cells"]
        if main_started[] > 0 && completed > 0
            status["estimated_remaining_seconds"] = max(0.0, (now - main_started[]) * (prod(shape) - completed) / completed)
        end
        if force || stage != last_stage[] || now - last_print[] >= 10
            remaining = get(status, "estimated_remaining_seconds", nothing)
            eta = remaining === nothing ? "" : "; ETA " * string(round(remaining; digits=1)) * " s"
            reader_event = get(status, "reader_progress", nothing)
            same_stage = reader_event !== nothing && reader_event.stage == stage
            layers = !same_stage ? "" : "; native layers " * string(reader_event.completed_layers) * "/" * string(reader_event.total_layers)
            println(stage, ": ", completed, "/", prod(shape), " cells", layers, "; elapsed ", round(now - began; digits=1), " s", eta)
            flush(stdout)
            write_json(joinpath(output, "status.json"), status)
            last_print[] = now
            last_stage[] = stage
        end
    end
    progress("density_prepass"; force=true)
    try
        reading_started = time()
        reader_stats = foreach_snapshot_batch(
            snapshot;
            x_range=selected_x,
            y_range=selected_y,
            physics=model.meta["constants"],
            batch_cells=batch_cells,
            max_cache_bytes=reader_cache_bytes,
            progress=event -> progress(String(event.stage); details=event),
        ) do tile
            main_started[] == 0 && (main_started[] = time())
            started = time()
            emission = calculate_emission(model, tile)
            timings["emission_seconds"] += time() - started
            started = time()
            add_batch!(accumulator, tile, emission)
            timings["products_seconds"] += time() - started
            started = time()
            add_expected_luminosity!(expected, emission, tile, opacity)
            expected_mass += sum(tile.density_g_cm3) * tile.cell_volume_cm3
            merge_emission_diagnostics!(diagnostics, emission.diagnostics)
            timings["independent_accounting_seconds"] += time() - started
            if save_cells
                started = time()
                save_batch_cells(output, accumulator.batch_count, tile, emission)
                timings["debug_save_seconds"] += time() - started
            end
            status["completed_cells"] = accumulator.processed_cells
            status["completed_batches"] = accumulator.batch_count
            progress("processing")
        end
        timings["streaming_seconds_including_compute"] = time() - reading_started
        progress("finalizing_products"; force=true)
        started = time()
        products = finish_products(accumulator)
        residuals = check_snapshot_products(products, expected, expected_mass, prod(shape))
        prepare_display!(products, geometry, model.meta)
        timings["finalization_seconds"] = time() - started
        source_identity(dataset, assets) == identity || error("Inputs or implementation changed during processing")
        NPZ.npzwrite(joinpath(output, "products.tmp.npz"), products)
        mv(joinpath(output, "products.tmp.npz"), joinpath(output, "products.npz"))
        report = Dict{String,Any}(
            "status" => "preparing", "processing_complete" => false, "raw_processing_complete" => true,
            "implementation" => "Julia/Mera streaming snapshot process", "schema_version" => 1,
            "full_snapshot" => shape == snapshot.original_shape,
            "dataset" => dataset, "jobname" => jobname, "snapshot_name" => basename(dataset),
            "physics" => model.meta, "identity" => identity,
            "x_index_range" => [first(selected_x) - 1, last(selected_x)],
            "y_index_range" => [first(selected_y) - 1, last(selected_y)],
            "shape" => collect(shape), "snapshot_shape" => collect(snapshot.original_shape),
            "cell_count" => prod(shape), "completed_batches" => accumulator.batch_count,
            "cell_width_cm" => collect(snapshot.cell_width_cm),
            "domain_left_cm" => collect(snapshot.domain_left_cm),
            "domain_right_cm" => collect(snapshot.domain_right_cm), "time_s" => snapshot.time_s,
            "reader_stats" => reader_stats, "timing" => timings,
            "threads" => threads, "batch_cells" => batch_cells,
            "reader_cache_budget_bytes" => reader_cache_bytes,
            "peak_rss_bytes" => Sys.maxrss(), "independent_total_mass_g" => expected_mass,
            "conservation_relative_residuals" => residuals, "emission_diagnostics" => diagnostics,
            "axis_order" => Dict("images" => ["x", "y", "line", "dust"], "spectra" => ["channel", "line", "dust", "regime"]),
            "dust_states" => ["intrinsic", "attenuated"], "regimes" => ["cold", "hot"],
            "plot_status" => make_plots ? "pending" : "not_requested",
        )
        write_json(joinpath(output, "report.json"), report)
        progress("preparing_saved_figures"; force=true)
        started = time()
        run_python_stage(python, "prepare_results.py", [output, "--image-downsample-factor", string(image_downsample_factor)])
        timings["figure_preparation_seconds"] = time() - started
        source_identity(dataset, assets) == identity || error("Inputs or implementation changed during preparation")
        report["processing_complete"] = true
        report["status"] = "completed"
        report["peak_rss_bytes"] = Sys.maxrss()
        timings["numerical_total_seconds"] = time() - began
        write_json(joinpath(output, "report.json"), report)
        status["processing_complete"] = true
        status["status"] = "completed"
        progress("numerical_complete"; force=true)
        if make_plots
            status["plot_status"] = "running"
            progress("rendering"; force=true)
            arguments = [output, "--format", figure_formats..., "--style", figure_styles..., "--image-downsample-factor", string(image_downsample_factor)]
            raw_luminosity && push!(arguments, "--raw-luminosity")
            started = time()
            run_python_stage(python, "plot_results.py", arguments)
            timings["rendering_seconds"] = time() - started
            status["plot_status"] = "completed"
            report["plot_status"] = "completed"
            write_json(joinpath(output, "report.json"), report)
        else
            status["plot_status"] = "not_requested"
        end
        progress("completed"; force=true)
        println("Completed snapshot products: ", output)
        return report
    catch exception
        if status["processing_complete"]
            status["plot_status"] = "failed"
            report["plot_status"] = "failed"
            report["plot_error"] = sprint(showerror, exception)
            write_json(joinpath(output, "report.json"), report)
        else
            status["status"] = "failed"
        end
        status["error"] = sprint(showerror, exception)
        status["elapsed_seconds"] = time() - began
        write_json(joinpath(output, "status.json"), status)
        rethrow()
    end
end
