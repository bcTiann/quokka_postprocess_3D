#!/usr/bin/env julia
using QuokkaProcess

function main(arguments)
    options = Dict{String,Any}(
        "threads" => Threads.nthreads(),
        "batch-cells" => 131_072,
        "reader-cache-mib" => 1024,
        "image-downsample-factor" => 1,
        "no-plot" => false,
        "save-cells" => false,
        "raw-luminosity" => false,
        "format" => ["png"],
        "style" => ["titled"],
    )
    integers = ("--threads", "--batch-cells", "--reader-cache-mib", "--image-downsample-factor")
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--help"
            println("""
Usage: julia --project=julia --threads=4 julia/bin/process.jl --dataset ABS_PATH --jobname NAME --assets DIR --output NEW_DIR [options]

The default processes the entire snapshot, prepares saved numerical figures,
and renders all standard plots. Inputs use the current fixed emission assets.

Options:
  --jobname NAME                   Required explicit simulation/job label.
  --x START STOP / --y START STOP   Optional 0-based half-open native ranges.
  --batch-cells N                   Emission batch target (default 131072).
  --reader-cache-mib N              Retained reader-array budget (default 1024).
  --threads N                      Product tasks; at most Julia launch threads.
  --python PATH                    Python with this repository installed.
  --no-plot                        Prepare numerical products without rendering.
  --format png pdf                 One or both formats (default png).
  --style titled untitled          One or both styles (default titled).
  --image-downsample-factor N      Prepare and select pixel sums (default 1).
  --raw-luminosity                 Plot saved luminosity instead of per-area spectra.
  --save-cells                     Save large per-batch verification arrays.

The reader budget excludes Mera temporary arrays and the Julia runtime. Existing
outputs are preserved; choose a new directory. z always spans the original box.
""")
            return
        elseif argument in ("--no-plot", "--save-cells", "--raw-luminosity")
            options[argument[3:end]] = true
        elseif argument in ("--x", "--y")
            index + 2 <= length(arguments) || error("$argument needs START STOP")
            start = parse(Int, arguments[index + 1])
            stop = parse(Int, arguments[index + 2])
            0 <= start < stop || error("Invalid $argument range")
            options[argument[3:end]] = (start + 1):stop
            index += 2
        elseif argument in ("--format", "--style")
            values = String[]
            while index < length(arguments) && !startswith(arguments[index + 1], "--")
                index += 1
                push!(values, arguments[index])
            end
            isempty(values) && error("$argument needs at least one value")
            options[argument[3:end]] = unique(values)
        elseif argument == "--launch-record"
            index < length(arguments) || error("Missing value for $argument")
            options["launch-record"] = QuokkaProcess.JSON3.read(arguments[index + 1], Dict{String,Any})
            index += 1
        elseif argument in ("--dataset", "--jobname", "--assets", "--output", "--python", integers...)
            index < length(arguments) || error("Missing value for $argument")
            value = arguments[index + 1]
            options[argument[3:end]] = argument in integers ? parse(Int, value) : value
            index += 1
        else
            error("Unknown argument: $argument")
        end
        index += 1
    end
    for name in ("dataset", "jobname", "assets", "output")
        haskey(options, name) || error("Required option: --$name")
    end
    isabspath(expanduser(options["dataset"])) || error("--dataset must be absolute")
    python = haskey(options, "python") ? expanduser(options["python"]) : QuokkaProcess.default_python()
    run_snapshot(
        dataset=expanduser(options["dataset"]),
        jobname=options["jobname"],
        assets=expanduser(options["assets"]),
        output=expanduser(options["output"]),
        x_range=get(options, "x", nothing),
        y_range=get(options, "y", nothing),
        threads=options["threads"],
        batch_cells=options["batch-cells"],
        reader_cache_bytes=options["reader-cache-mib"] * 1024^2,
        python=python,
        make_plots=!options["no-plot"],
        figure_formats=options["format"],
        figure_styles=options["style"],
        image_downsample_factor=options["image-downsample-factor"],
        raw_luminosity=options["raw-luminosity"],
        save_cells=options["save-cells"],
        launch_configuration=get(options, "launch-record", nothing),
    )
end

main(ARGS)
