"""Stable weighted moments, including an optional within-cell variance.

`weight` is luminosity [erg/s] for a line, or mass [g] for a gas phase.
`mean` is LOS velocity [km/s]; `second` is the weighted centered second
moment [weight * (km/s)^2]. Empty populations retain zero internal state.
"""
mutable struct ProductMoments
    weight::Float64
    mean::Float64
    second::Float64
end

ProductMoments() = ProductMoments(0.0, 0.0, 0.0)

function add_product_moment!(moment, weight, velocity, variance=0.0)
    weight == 0.0 && return moment
    combined_weight = moment.weight + weight
    difference = velocity - moment.mean
    fraction = weight / combined_weight
    moment.second += weight * variance + difference^2 * moment.weight * fraction
    moment.mean += difference * fraction
    moment.weight = combined_weight
    return moment
end

function merge_product_moments!(moment, other)
    other.weight == 0.0 && return moment
    combined_weight = moment.weight + other.weight
    difference = other.mean - moment.mean
    fraction = other.weight / combined_weight
    moment.second += other.second + difference^2 * moment.weight * fraction
    moment.mean += difference * fraction
    moment.weight = combined_weight
    return moment
end

product_centroid(moment) = moment.weight > 0.0 ? moment.mean : NaN
product_sigma(moment) = moment.weight > 0.0 ? sqrt(moment.second / moment.weight) : NaN

function new_product_accumulator(channel_count, line_count, phase_count)
    return (
        spectra=zeros(channel_count, line_count, 2, 2),
        line_moments=reshape([ProductMoments() for _ in 1:line_count*4], line_count, 2, 2),
        line_counts=zeros(Int64, line_count, 2),
        missing_lines=zeros(Int64, line_count),
        missing_line_mass=zeros(line_count),
        available_line_mass=zeros(line_count, 2),
        processed_cell_count=zeros(Int64, 1),
        total_cell_mass=zeros(1),
        regime_cell_count=zeros(Int64, 2),
        regime_cell_mass=zeros(2),
        phase_full=[ProductMoments() for _ in 1:phase_count],
        phase_window=[ProductMoments() for _ in 1:phase_count],
        phase_counts=zeros(Int64, phase_count),
        phase_window_counts=zeros(Int64, phase_count),
        phase_histogram=zeros(channel_count, phase_count),
        below_counts=zeros(Int64, phase_count),
        above_counts=zeros(Int64, phase_count),
        below_mass=zeros(phase_count),
        above_mass=zeros(phase_count),
        missing_temperature_count=zeros(Int64, 1),
        missing_temperature_mass=zeros(1),
        scratch=(
            luminosity=zeros(line_count, 2),
            widths=zeros(line_count),
            shared=zeros(Int, line_count),
            done=falses(line_count),
        ),
    )
end

"""Accumulate unbroadened gas velocities; the rightmost histogram edge is included."""
function add_phase_cell!(state, phase, velocity, mass, edges)
    total_group = length(state.phase_full)
    for group in (phase, total_group)
        state.phase_counts[group] += 1
        add_product_moment!(state.phase_full[group], mass, velocity)
        if velocity < first(edges)
            state.below_counts[group] += 1
            state.below_mass[group] += mass
        elseif velocity > last(edges)
            state.above_counts[group] += 1
            state.above_mass[group] += mass
        else
            channel = min(searchsortedlast(edges, velocity), length(edges) - 1)
            state.phase_histogram[channel, group] += mass
            state.phase_window_counts[group] += 1
            add_product_moment!(state.phase_window[group], mass, velocity)
        end
    end
end

"""Return the rounded Float64 erf, avoiding special-function work in saturated tails.

The Gaussian-channel formula is unchanged: at |argument| >= 8, erf is already
rounded to +/-1 in Float64. This is not a finite-sigma physical truncation.
"""
@inline function product_erf(argument::Float64)
    argument >= 8.0 && return 1.0
    argument <= -8.0 && return -1.0
    return erf(argument)
end

"""Keep channels whose two erf arguments are not on the same saturated tail.

`edges` are increasing velocities [km/s]; `denominator` is sqrt(2) times
the thermal width [km/s]. Search the original Float64 quotient, since
inverting the threshold as velocity +/- 8 * denominator can round differently.
Only exactly zero contributions from `product_erf` are omitted.
"""
function gaussian_channel_range(edges, velocity, denominator)
    channel_count = length(edges) - 1
    isfinite(denominator) && denominator > 0.0 || return 1:channel_count
    first_channel = 1
    last_channel = channel_count

    if (first(edges) - velocity) / denominator <= -8.0
        lower = 1
        upper = length(edges) + 1
        while lower + 1 < upper
            middle = lower + (upper - lower) ÷ 2
            if (edges[middle] - velocity) / denominator <= -8.0
                lower = middle
            else
                upper = middle
            end
        end
        first_channel = lower
    end

    if (last(edges) - velocity) / denominator >= 8.0
        lower = 0
        upper = length(edges)
        while lower + 1 < upper
            middle = lower + (upper - lower) ÷ 2
            if (edges[middle] - velocity) / denominator >= 8.0
                upper = middle
            else
                lower = middle
            end
        end
        last_channel = upper - 1
    end
    return first_channel:last_channel
end

"""Fuse Gaussian channel probabilities with all same-width line luminosities.

Each task owns its spectra and scratch. A cell's two dust states and lines
with identical thermal widths reuse each erf difference. No channel-cell
probability matrix or nested thread pool is created.
"""
function add_cell_spectra!(
    spectra;
    luminosity,
    widths,
    branch,
    velocity,
    edges,
    shared,
    done,
)
    fill!(done, false)
    line_count = size(luminosity, 1)
    for first_line in 1:line_count
        (done[first_line] || luminosity[first_line, 1] == 0.0) && continue
        shared_count = 0
        for line in first_line:line_count
            if !done[line] && luminosity[line, 1] > 0.0 && widths[line] == widths[first_line]
                shared_count += 1
                shared[shared_count] = line
                done[line] = true
            end
        end
        denominator = sqrt(2.0) * widths[first_line]
        channels = gaussian_channel_range(edges, velocity, denominator)
        isempty(channels) && continue
        previous = product_erf((edges[first(channels)] - velocity) / denominator)
        for channel in channels
            current = product_erf((edges[channel + 1] - velocity) / denominator)
            fraction_per_kms = 0.5 * (current - previous) / (edges[channel + 1] - edges[channel])
            previous = current
            fraction_per_kms == 0.0 && continue
            # Product shapes are checked before accumulation; task-owned scratch
            # contains only line indices from the validated luminosity matrix.
            @inbounds for position in 1:shared_count
                line = shared[position]
                spectra[channel, line, 1, branch] += fraction_per_kms * luminosity[line, 1]
                spectra[channel, line, 2, branch] += fraction_per_kms * luminosity[line, 2]
            end
        end
    end
end

function accumulate_product_columns!(images, columns, tile, emission, parameters; state=nothing)
    nx, ny, nz = size(tile.density_g_cm3)
    column_count = nx * ny
    line_count = length(parameters.masses_g)
    channel_count = length(parameters.edges) - 1
    phase_count = length(parameters.phase_bounds) + 2
    if isnothing(state)
        state = new_product_accumulator(channel_count, line_count, phase_count)
    end
    luminosity = state.scratch.luminosity
    widths = state.scratch.widths
    shared = state.scratch.shared
    done = state.scratch.done
    image_columns = reshape(images, column_count, line_count, 2)

    # Tasks own disjoint x-y columns; z planes retain contiguous x-fastest access.
    for z in 1:nz
        for column in columns
            cell = column + (z - 1) * column_count
            velocity = tile.velocity_z_kms[cell]
            foreground = tile.foreground_NH_cm2[cell]
            mass = tile.density_g_cm3[cell] * tile.cell_volume_cm3
            isfinite(velocity) && abs(velocity) < parameters.light_speed ||
                throw(ArgumentError("LOS velocity must be finite with abs(v) < c"))
            isfinite(foreground) && foreground >= 0.0 ||
                throw(ArgumentError("Foreground hydrogen column must be finite and nonnegative"))
            isfinite(mass) && mass > 0.0 ||
                throw(ArgumentError("Cell mass must be finite and positive"))
            cold = emission.cold_cells[cell]
            branch = cold ? 1 : 2
            state.processed_cell_count[1] += 1
            state.total_cell_mass[1] += mass
            state.regime_cell_count[branch] += 1
            state.regime_cell_mass[branch] += mass
            phase_temperature = cold ? emission.despotic_temperature_K[cell] : tile.temperature_quokka_K[cell]
            if isfinite(phase_temperature) && phase_temperature > 0.0
                phase = searchsortedlast(parameters.phase_bounds, phase_temperature) + 1
                add_phase_cell!(state, phase, velocity, mass, parameters.edges)
            else
                state.missing_temperature_count[1] += 1
                state.missing_temperature_mass[1] += mass
            end

            fill!(luminosity, 0.0)
            for line in 1:line_count
                emissivity = emission.emissivity_erg_s_cm3[cell, line]
                if !isfinite(emissivity)
                    state.missing_lines[line] += 1
                    state.missing_line_mass[line] += mass
                    continue
                end
                temperature = emission.temperature_K[cell, line]
                emissivity >= 0.0 || throw(ArgumentError("Available emissivity must be nonnegative"))
                isfinite(temperature) && temperature >= 0.0 ||
                    throw(ArgumentError("Available line temperature must be finite and nonnegative"))
                intrinsic = emissivity * tile.cell_volume_cm3
                isfinite(intrinsic) || throw(ArgumentError("Cell luminosity overflowed"))
                attenuated = intrinsic * exp(-parameters.opacities[line] * foreground)
                luminosity[line, 1] = intrinsic
                luminosity[line, 2] = attenuated
                width = sqrt(parameters.boltzmann * temperature / parameters.masses_g[line]) / 1.0e5
                width *= 1.0 - velocity / parameters.light_speed
                width = max(width, parameters.minimum_width)
                widths[line] = width
                state.line_counts[line, branch] += 1
                state.available_line_mass[line, branch] += mass
                image_columns[column, line, 1] += intrinsic
                image_columns[column, line, 2] += attenuated
                add_product_moment!(state.line_moments[line, 1, branch], intrinsic, velocity, width^2)
                add_product_moment!(state.line_moments[line, 2, branch], attenuated, velocity, width^2)
            end
            add_cell_spectra!(
                state.spectra;
                luminosity=luminosity,
                widths=widths,
                branch=branch,
                velocity=velocity,
                edges=parameters.edges,
                shared=shared,
                done=done,
            )
        end
    end
    return state
end

function merge_product_accumulators!(state, increment)
    for field in (
        :spectra, :line_counts, :missing_lines, :missing_line_mass, :available_line_mass,
        :processed_cell_count, :total_cell_mass, :regime_cell_count, :regime_cell_mass,
        :phase_counts, :phase_window_counts,
        :phase_histogram, :below_counts, :above_counts, :below_mass, :above_mass,
        :missing_temperature_count, :missing_temperature_mass,
    )
        getproperty(state, field) .+= getproperty(increment, field)
    end
    for index in eachindex(state.line_moments)
        merge_product_moments!(state.line_moments[index], increment.line_moments[index])
    end
    for group in eachindex(state.phase_full)
        merge_product_moments!(state.phase_full[group], increment.phase_full[group])
        merge_product_moments!(state.phase_window[group], increment.phase_window[group])
    end
    return state
end

"""Compute channel-center moments, matching the saved-window interpretation.

These are moments of the binned line. Complete analytic Gaussian moments are
stored separately and include both thermal variance and out-of-window light.
"""
function channel_product_moments(spectra, edges)
    trailing_shape = size(spectra)[2:end]
    channel_count = size(spectra, 1)
    profiles = reshape(spectra, channel_count, :)
    captured = zeros(size(profiles, 2))
    centroid = fill(NaN, size(profiles, 2))
    sigma = fill(NaN, size(profiles, 2))
    for profile in axes(profiles, 2)
        moment = ProductMoments()
        for channel in 1:channel_count
            weight = profiles[channel, profile] * (edges[channel + 1] - edges[channel])
            center = 0.5 * (edges[channel + 1] + edges[channel])
            add_product_moment!(moment, weight, center)
        end
        captured[profile] = moment.weight
        centroid[profile] = product_centroid(moment)
        sigma[profile] = product_sigma(moment)
    end
    return reshape(captured, trailing_shape), reshape(centroid, trailing_shape), reshape(sigma, trailing_shape)
end

function normalized_product_profiles(profiles)
    normalized = zeros(size(profiles))
    matrix = reshape(profiles, size(profiles, 1), :)
    output = reshape(normalized, size(profiles, 1), :)
    for profile in axes(matrix, 2)
        peak = maximum(view(matrix, :, profile))
        if peak > 0.0
            output[:, profile] .= view(matrix, :, profile) ./ peak
        end
    end
    return normalized
end

function build_product_output(images, state, tile, parameters)
    line_count = size(images, 3)
    moments = state.line_moments
    input_light = map(moment -> moment.weight, moments)
    full_centroid_by_regime = map(product_centroid, moments)
    full_sigma_by_regime = map(product_sigma, moments)
    full_second_by_regime = map(moment -> moment.second, moments)
    combined = reshape([ProductMoments() for _ in 1:line_count*2], line_count, 2)
    for dust in 1:2
        for line in 1:line_count
            merge_product_moments!(combined[line, dust], moments[line, dust, 1])
            merge_product_moments!(combined[line, dust], moments[line, dust, 2])
        end
    end
    captured, window_centroid_by_regime, window_sigma_by_regime = channel_product_moments(
        state.spectra,
        parameters.edges,
    )
    total_spectra = dropdims(sum(state.spectra; dims=4); dims=4)
    _, window_centroid, window_sigma = channel_product_moments(
        total_spectra,
        parameters.edges,
    )
    image_totals = dropdims(sum(images; dims=(1, 2)); dims=(1, 2))
    color_limits = zeros(line_count, 2)
    for line in 1:line_count
        peak = maximum(view(images, :, :, line, :))
        color_limits[line, 1] = peak / 1.0e5
        color_limits[line, 2] = peak
    end
    pixel_area = tile.cell_width_cm[1] * tile.cell_width_cm[2]
    projected_area = size(images, 1) * size(images, 2) * pixel_area
    phase_full = state.phase_full
    phase_window = state.phase_window
    global_mean = product_centroid(last(phase_full))
    phase_mass = map(moment -> moment.weight, phase_full)
    phase_count = state.phase_counts
    output = Dict{String,Any}(
        "schema_version" => Int64[1],
        "cell_count" => copy(state.processed_cell_count),
        "total_cell_mass_g" => copy(state.total_cell_mass),
        "cell_counts_by_temperature_regime" => copy(state.regime_cell_count),
        "cell_mass_by_temperature_regime_g" => copy(state.regime_cell_mass),
        "native_xy_shape" => Int64[size(images, 1), size(images, 2)],
        "image_xy_origin" => Int64[first(tile.x_range) - 1, first(tile.y_range) - 1],
        "cell_width_cm" => collect(Float64, tile.cell_width_cm),
        "cell_volume_cm3" => Float64[tile.cell_volume_cm3],
        "projected_area_cm2" => Float64[projected_area],
        "velocity_edges_kms" => copy(parameters.edges),
        "velocity_centers_kms" => 0.5 .* (parameters.edges[1:end-1] .+ parameters.edges[2:end]),
        "images_luminosity_erg_s" => images,
        "image_total_luminosity_erg_s" => image_totals,
        "image_color_limits_erg_s" => color_limits,
        "image_is_hidden" => UInt8.(images .<= 0.0),
        "spectra_dL_dv_erg_s_per_kms" => state.spectra,
        "spectra_total_dL_dv_erg_s_per_kms" => total_spectra,
        "spectra_total_peak_normalized" => normalized_product_profiles(total_spectra),
        "spectra_total_surface_luminosity_erg_s_cm2_per_kms" => total_spectra ./ projected_area,
        "input_luminosity_erg_s" => input_light,
        "captured_luminosity_erg_s" => captured,
        "outside_velocity_luminosity_erg_s" => max.(0.0, input_light .- captured),
        "full_line_luminosity_erg_s" => map(moment -> moment.weight, combined),
        "line_centroid_full_kms" => map(product_centroid, combined),
        "line_sigma_full_kms" => map(product_sigma, combined),
        "line_centroid_full_by_regime_kms" => full_centroid_by_regime,
        "line_sigma_full_by_regime_kms" => full_sigma_by_regime,
        "line_centered_second_moment_full_by_regime" => full_second_by_regime,
        "line_centroid_window_kms" => window_centroid,
        "line_sigma_window_kms" => window_sigma,
        "line_centroid_window_by_regime_kms" => window_centroid_by_regime,
        "line_sigma_window_by_regime_kms" => window_sigma_by_regime,
        "cell_counts_by_regime" => state.line_counts,
        "missing_emissivity_cells" => state.missing_lines,
        "missing_emissivity_mass_g" => state.missing_line_mass,
        "available_emissivity_mass_by_regime_g" => state.available_line_mass,
        "phase_boundaries_K" => copy(parameters.phase_bounds),
        "phase_histogram_mass_g" => state.phase_histogram,
        "phase_histogram_peak_normalized" => normalized_product_profiles(state.phase_histogram),
        "phase_cell_count" => phase_count,
        "phase_mass_g" => phase_mass,
        "phase_centroid_kms" => map(product_centroid, phase_full),
        "phase_sigma_internal_kms" => map(product_sigma, phase_full),
        "phase_centered_second_moment" => map(moment -> moment.second, phase_full),
        "phase_sigma_about_global_mean_kms" => [
            moment.weight > 0.0 ? hypot(product_sigma(moment), moment.mean - global_mean) : NaN
            for moment in phase_full
        ],
        "phase_window_cell_count" => state.phase_window_counts,
        "phase_window_mass_g" => map(moment -> moment.weight, phase_window),
        "phase_window_centroid_kms" => map(product_centroid, phase_window),
        "phase_window_sigma_internal_kms" => map(product_sigma, phase_window),
        "phase_window_centered_second_moment" => map(moment -> moment.second, phase_window),
        "phase_window_sigma_about_global_mean_kms" => [
            moment.weight > 0.0 ? hypot(product_sigma(moment), moment.mean - global_mean) : NaN
            for moment in phase_window
        ],
        "phase_below_window_count" => state.below_counts,
        "phase_above_window_count" => state.above_counts,
        "phase_below_window_mass_g" => state.below_mass,
        "phase_above_window_mass_g" => state.above_mass,
        "phase_missing_temperature_count" => state.missing_temperature_count,
        "phase_missing_temperature_mass_g" => state.missing_temperature_mass,
    )
    return output
end

"""Check independent image, Gaussian and gas-phase accounting at relative 1e-10."""
function check_product_accounting(output)
    cell_count = only(output["cell_count"])
    counts = output["cell_counts_by_regime"]
    missing = output["missing_emissivity_cells"]
    all(vec(sum(counts; dims=2)) .+ missing .== cell_count) || error("Line cell accounting failed")
    total_mass = only(output["total_cell_mass_g"])
    isfinite(total_mass) && total_mass > 0.0 || error("Independent gas mass is invalid")
    line_mass = vec(sum(output["available_emissivity_mass_by_regime_g"]; dims=2))
    line_mass .+= output["missing_emissivity_mass_g"]
    all(isapprox.(line_mass, total_mass; rtol=1.0e-10, atol=0.0)) || error("Line missing/available mass accounting failed")
    sum(output["cell_counts_by_temperature_regime"]) == cell_count || error("Temperature-regime cell accounting failed")
    isapprox(sum(output["cell_mass_by_temperature_regime_g"]), total_mass; rtol=1.0e-10, atol=0.0) || error("Temperature-regime mass accounting failed")
    input_light = output["input_luminosity_erg_s"]
    total_light = dropdims(sum(input_light; dims=3); dims=3)
    for actual in (output["image_total_luminosity_erg_s"], output["full_line_luminosity_erg_s"])
        all(isapprox.(actual, total_light; rtol=1.0e-10, atol=0.0)) || error("Image/full-line luminosity conservation failed")
    end
    all(isfinite, input_light) || error("Accumulated line luminosity overflowed")
    all(isfinite, output["spectra_dL_dv_erg_s_per_kms"]) || error("Accumulated spectra overflowed")
    recovered = output["captured_luminosity_erg_s"] .+ output["outside_velocity_luminosity_erg_s"]
    all(isapprox.(recovered, input_light; rtol=1.0e-10, atol=0.0)) || error("Gaussian window accounting failed")
    phase_counts = output["phase_cell_count"]
    phase_mass = output["phase_mass_g"]
    all(isfinite, phase_mass) || error("Accumulated phase mass overflowed")
    sum(phase_counts[1:end-1]) == last(phase_counts) || error("Phase cell accounting failed")
    last(phase_counts) + only(output["phase_missing_temperature_count"]) == cell_count || error("Mixed-temperature cell accounting failed")
    recovered_total_mass = last(phase_mass) + only(output["phase_missing_temperature_mass_g"])
    isapprox(recovered_total_mass, total_mass; rtol=1.0e-10, atol=0.0) || error("Independent gas mass accounting failed")
    isapprox(sum(phase_mass[1:end-1]), last(phase_mass); rtol=1.0e-10, atol=0.0) || error("Phase mass accounting failed")
    histogram = output["phase_histogram_mass_g"]
    all(isapprox.(vec(sum(histogram[:, 1:end-1]; dims=2)), histogram[:, end]; rtol=1.0e-10, atol=0.0)) || error("Phase histogram accounting failed")
    window_counts = output["phase_window_cell_count"]
    below_counts = output["phase_below_window_count"]
    above_counts = output["phase_above_window_count"]
    all(window_counts .+ below_counts .+ above_counts .== phase_counts) || error("Gas velocity-window cell accounting failed")
    window_mass = vec(sum(histogram; dims=1))
    recovered_mass = window_mass .+ output["phase_below_window_mass_g"] .+ output["phase_above_window_mass_g"]
    all(isapprox.(recovered_mass, phase_mass; rtol=1.0e-10, atol=0.0)) || error("Gas velocity-window mass accounting failed")
    return output
end

"""Validate one x-fastest cell batch and its cell-by-line emission matrices."""
function check_product_batch(tile, emission, line_count)
    shape = size(tile.density_g_cm3)
    length(shape) == 3 && all(shape .> 0) || throw(ArgumentError("Expected a nonempty 3-D tile"))
    cell_count = length(tile.density_g_cm3)
    length(tile.x_range) == shape[1] || throw(DimensionMismatch("Native x range length"))
    length(tile.y_range) == shape[2] || throw(DimensionMismatch("Native y range length"))
    size(emission.emissivity_erg_s_cm3) == (cell_count, line_count) || throw(DimensionMismatch("Line emissivity shape"))
    size(emission.temperature_K) == (cell_count, line_count) || throw(DimensionMismatch("Line temperature shape"))
    length(emission.cold_cells) == cell_count || throw(DimensionMismatch("Cold-cell mask length"))
    length(emission.despotic_temperature_K) == cell_count || throw(DimensionMismatch("DESPOTIC temperature length"))
    for field in (:temperature_quokka_K, :foreground_NH_cm2, :velocity_z_kms)
        size(getproperty(tile, field)) == shape || throw(DimensionMismatch("Tile field $field shape"))
    end
    isfinite(tile.cell_volume_cm3) && tile.cell_volume_cm3 > 0.0 || throw(ArgumentError("Invalid cell volume"))
    all(isfinite, tile.cell_width_cm) && all(tile.cell_width_cm .> 0.0) || throw(ArgumentError("Invalid cell widths"))
    return nothing
end

"""Copy and validate the small physical inputs used by all product batches."""
function product_parameters(meta)
    line_count = length(meta["line_keys"])
    line_count > 0 || throw(ArgumentError("At least one emission line is required"))
    constants = meta["constants"]
    parameters = (
        edges=Float64.(meta["velocity_edges_kms"]),
        masses_g=Float64.(meta["masses_g"]),
        opacities=Float64.(meta["dust_cross_sections_cm2_H"]),
        phase_bounds=Float64.(meta["phase_boundaries_K"]),
        boltzmann=Float64(constants["boltzmann_erg_K"]),
        light_speed=Float64(constants["speed_of_light_kms"]),
        minimum_width=Float64(meta["minimum_gaussian_width_kms"]),
    )
    if length(parameters.edges) < 2 || !all(isfinite, parameters.edges) || !all(diff(parameters.edges) .> 0.0)
        throw(ArgumentError("Velocity edges must be finite and increasing"))
    end
    if length(parameters.masses_g) != line_count || !all(isfinite, parameters.masses_g) || !all(parameters.masses_g .> 0.0)
        throw(ArgumentError("Invalid emitter masses"))
    end
    if length(parameters.opacities) != line_count || !all(isfinite, parameters.opacities) || !all(parameters.opacities .>= 0.0)
        throw(ArgumentError("Invalid dust opacities"))
    end
    if !all(isfinite, parameters.phase_bounds) || !all(diff(parameters.phase_bounds) .> 0.0)
        throw(ArgumentError("Phase boundaries must be finite and increasing"))
    end
    for value in (parameters.boltzmann, parameters.light_speed, parameters.minimum_width)
        isfinite(value) && value > 0.0 || throw(ArgumentError("Product constants must be finite and positive"))
    end
    return parameters
end

"""
    accumulate_products(tile, emission, meta; threads=Threads.nthreads())

Calculate LOS-z products from one complete-z tile and return numeric arrays.
Tile fields are Float64 `(Nx,Ny,Nz)` with x fastest; `cell_width_cm` is a
3-tuple, `cell_volume_cm3` is scalar, and x/y ranges use native 1-based indices.
Emission emissivity and line temperature are `(Nx*Ny*Nz,L)`; cold cells and
DESPOTIC temperature follow `vec(tile.density_g_cm3)` order. Missing line
emissivities are excluded per line; finite physical zeros remain counted.

Schema 1 axis orders:
- images: `(x,y,line,dust)`, [erg/s per native pixel].
- spectra: `(channel,line,dust,regime)`, [erg/s/(km/s)].
- branch luminosities and line moments: `(line,dust,regime)`.
- total luminosities and line moments: `(line,dust)`.
- line cell counts: `(line,regime)`; missing counts: `(line,)`.
- available line mass: `(line,regime)` [g]; missing line mass: `(line,)` [g].
- phase histograms: `(channel,phase)` [g]; phase statistics: `(phase,)`.
Dust order is intrinsic, attenuated. Regimes are cold, hot, defined by the
emission input. Phases are CNM,UNM,WNM,WIM,HIM,total; thresholds belong to the
hotter phase. Empty centroids/dispersions are NaN, never an artificial zero.
Scalar quantities are length-one arrays for portable NPZ writing. Strings
and source identities belong to the surrounding JSON metadata.

Full moments include each complete Gaussian, with the adopted `(1-v/c)`
thermal width and minimum width. Window moments use channel centers and
saved channel-integrated luminosities. Gas phases use unbroadened velocities
and mixed temperature independently of line missingness. The kernel stores
no cell-by-channel matrix. Each task owns disjoint image columns and private
small spectra/moments, merged in task order after a single parallel layer.
"""
function accumulate_products(tile, emission, meta; threads=Threads.nthreads())
    line_count = length(meta["line_keys"])
    check_product_batch(tile, emission, line_count)
    parameters = product_parameters(meta)
    shape = size(tile.density_g_cm3)
    images = zeros(shape[1], shape[2], line_count, 2)
    column_count = shape[1] * shape[2]
    task_count = min(max(Int(threads), 1), Threads.nthreads(), column_count)
    ranges = [
        fld((task - 1) * column_count, task_count) + 1:fld(task * column_count, task_count)
        for task in 1:task_count
    ]
    if task_count == 1
        state = accumulate_product_columns!(images, only(ranges), tile, emission, parameters)
    else
        tasks = [
            Threads.@spawn accumulate_product_columns!(images, columns, tile, emission, parameters)
            for columns in ranges
        ]
        # Wait for every task even when one fails, so no task outlives its tile.
        states = Vector{Any}(undef, task_count)
        failure = nothing
        for (index, task) in enumerate(tasks)
            try
                states[index] = fetch(task)
            catch exception
                failure = exception
            end
        end
        isnothing(failure) || throw(failure)
        state = states[1]
        for increment in states[2:end]
            merge_product_accumulators!(state, increment)
        end
    end
    output = build_product_output(images, state, tile, parameters)
    return check_product_accounting(output)
end
