"""Fixed lookup tables and constants; cell arrays are never retained here."""
Base.@kwdef struct EmissionModel
    meta::Dict{String,Any}
    despotic_axes::NTuple{3,Vector{Float64}}
    despotic_log_axes::NTuple{3,Vector{Float64}}
    despotic_values::Array{Float64,4}
    cloudy_log_axes::NTuple{3,Vector{Float64}}
    cloudy_values::Array{Float64,4}
    cloudy_log_values::Array{Float64,4}
    cloudy_failures::BitArray{4}
    cloudy_zeros::BitArray{4}
    line_indices::Dict{String,Int}
    despotic_boundary_rtol::Float64
    cloudy_boundary_rtol::Float64
    cloudy_touch_eps::Float64
    branch_temperature_K::Float64
    halpha_photon_energy_erg::Float64
    hi21_emission_per_atom::Float64
end

"""Load numeric NPZ assets exported from the adopted, unchanged Python tables."""
function load_emission_model(asset_dir::AbstractString)
    meta = JSON3.read(read(joinpath(asset_dir, "physics.json"), String), Dict{String,Any})
    meta["schema_version"] == 1 || error("Unsupported emission asset schema")
    tables_path = joinpath(asset_dir, "tables.npz")
    digest = open(tables_path, "r") do source
        bytes2hex(SHA.sha256(source))
    end
    digest == meta["numerical_assets_sha256"] || error("Emission asset hash mismatch")
    arrays = NPZ.npzread(tables_path)
    despotic_axes = (
        Vector{Float64}(arrays["despotic_nH_cm3"]),
        Vector{Float64}(arrays["despotic_NH_cm2"]),
        Vector{Float64}(arrays["despotic_dVdr_s"]),
    )
    cloudy_axes = (
        Vector{Float64}(arrays["cloudy_log_NH_cm2"]),
        Vector{Float64}(arrays["cloudy_log_nH_cm3"]),
        Vector{Float64}(arrays["cloudy_log_T_K"]),
    )
    for axis in (despotic_axes..., cloudy_axes...)
        length(axis) >= 2 || error("Interpolation axes require at least two nodes")
        all(isfinite, axis) || error("Nonfinite interpolation axis")
        all(diff(axis) .> 0.0) || error("Interpolation axes must increase strictly")
    end
    all(axis -> all(axis .> 0.0), despotic_axes) || error("DESPOTIC axes must be positive")
    despotic_values = Array{Float64,4}(arrays["despotic_values"])
    cloudy_values = Array{Float64,4}(arrays["cloudy_emissivity_per_nH2"])
    cloudy_log_values = Array{Float64,4}(arrays["cloudy_log_emissivity_per_nH2"])
    cloudy_failures = BitArray(arrays["cloudy_failure_mask"])
    cloudy_zeros = BitArray(arrays["cloudy_zero_mask"])
    size(despotic_values) == (length.(despotic_axes)..., 7) || error("DESPOTIC shape mismatch")
    cloudy_shape = (length.(cloudy_axes)..., 8)
    for values in (cloudy_values, cloudy_log_values, cloudy_failures, cloudy_zeros)
        size(values) == cloudy_shape || error("Cloudy shape mismatch")
    end
    all(isfinite, cloudy_values) || error("Cloudy linear coefficients must be finite")
    all(cloudy_values .>= 0.0) || error("Cloudy coefficients must be nonnegative")
    cloudy_zeros == ((cloudy_values .== 0.0) .& .!cloudy_failures) || error("Cloudy zero mask mismatch")
    all(cloudy_values[cloudy_failures] .== 0.0) || error("Cloudy failure placeholders must be zero")
    # Preserve finite historical log sentinels. Failed/zero support is checked
    # separately before these finite interpolation placeholders are used.
    replace!(value -> isfinite(value) ? value : 0.0, cloudy_log_values)
    line_keys = String.(meta["line_keys"])
    expected_atomic = Set(["cii", "halpha", "hi21", "ciii_977", "ciii_1907", "ciii_1909", "civ_1548", "civ_1551"])
    length(line_keys) == 10 || error("Expected ten emission lines")
    Set(line_keys[1:8]) == expected_atomic || error("Unexpected Cloudy line names")
    line_keys[9:10] == ["co10", "co21"] || error("Unexpected CO line order")
    line_indices = Dict(key => index for (index, key) in enumerate(line_keys))
    interpolation = meta["interpolation"]
    hydrogen = meta["analytic_hydrogen"]
    physical = meta["constants"]
    hi21_emission_per_atom = 0.75 * hydrogen["hi21_spontaneous_rate_s"] * physical["planck_erg_s"] * hydrogen["hi21_frequency_Hz"]
    return EmissionModel(
        meta=meta,
        despotic_axes=despotic_axes,
        despotic_log_axes=map(axis -> log10.(axis), despotic_axes),
        despotic_values=despotic_values,
        cloudy_log_axes=cloudy_axes,
        cloudy_values=cloudy_values,
        cloudy_log_values=cloudy_log_values,
        cloudy_failures=cloudy_failures,
        cloudy_zeros=cloudy_zeros,
        line_indices=line_indices,
        despotic_boundary_rtol=interpolation["despotic_boundary_rtol"],
        cloudy_boundary_rtol=interpolation["cloudy_log_boundary_rtol"],
        cloudy_touch_eps=interpolation["cloudy_touch_eps"],
        branch_temperature_K=meta["emission_temperature_boundary_K"],
        halpha_photon_energy_erg=hydrogen["halpha_photon_energy_erg"],
        hi21_emission_per_atom=hi21_emission_per_atom,
    )
end

"""Find a lower node and the fractional position on an increasing table axis."""
function interpolation_bracket(axis::Vector{Float64}, coordinate::Float64)
    lower = clamp(searchsortedlast(axis, coordinate), 1, length(axis) - 1)
    fraction = (coordinate - axis[lower]) / (axis[lower + 1] - axis[lower])
    return (lower, fraction)
end

"""Interpolate one field in three dimensions, including IEEE NaN propagation.

DESPOTIC follows the current SciPy linear interpolator: all eight corners take
part, so even NaN * 0 remains NaN at an exact face/node beside missing support.
"""
function trilinear_value(values::Array{Float64,4}, brackets, field::Int)
    value = 0.0
    for first_side in 0:1
        first_index = brackets[1][1] + first_side
        first_weight = first_side == 0 ? 1.0 - brackets[1][2] : brackets[1][2]
        for second_side in 0:1
            second_index = brackets[2][1] + second_side
            second_weight = second_side == 0 ? 1.0 - brackets[2][2] : brackets[2][2]
            for third_side in 0:1
                third_index = brackets[3][1] + third_side
                third_weight = third_side == 0 ? 1.0 - brackets[3][2] : brackets[3][2]
                weight = first_weight * second_weight * third_weight
                value += values[first_index, second_index, third_index, field] * weight
            end
        end
    end
    return value
end

"""Check physical DESPOTIC coverage before permitting tiny endpoint clipping."""
function despotic_coordinate(value::Float64, axis::Vector{Float64}, tolerance::Float64, name::String)
    if !isfinite(value) || value <= 0.0 || value < first(axis) * (1.0 - tolerance) || value > last(axis) * (1.0 + tolerance)
        throw(DomainError(value, "DESPOTIC $name outside the table domain"))
    end
    return clamp(value, first(axis), last(axis))
end

"""Require Cloudy nH/T coverage, then clip only logarithmic endpoint roundoff."""
function cloudy_coordinate(value::Float64, axis::Vector{Float64}, tolerance::Float64, name::String)
    allowed_error = tolerance * max(1.0, abs(first(axis)), abs(last(axis)))
    if value < first(axis) - allowed_error || value > last(axis) + allowed_error
        throw(DomainError(value, "Cloudy $name outside the table domain"))
    end
    return clamp(value, first(axis), last(axis))
end

"""Inspect one line's zero corners and total failed-node interpolation weight."""
function cloudy_support(model::EmissionModel, brackets, line::Int)
    zero_support = false
    failed_weight = 0.0
    for first_side in 0:1
        first_index = brackets[1][1] + first_side
        first_weight = first_side == 0 ? 1.0 - brackets[1][2] : brackets[1][2]
        for second_side in 0:1
            second_index = brackets[2][1] + second_side
            second_weight = second_side == 0 ? 1.0 - brackets[2][2] : brackets[2][2]
            for third_side in 0:1
                third_index = brackets[3][1] + third_side
                third_weight = third_side == 0 ? 1.0 - brackets[3][2] : brackets[3][2]
                weight = first_weight * second_weight * third_weight
                zero_support |= model.cloudy_zeros[first_index, second_index, third_index, line] && weight > model.cloudy_touch_eps
                failed_weight += model.cloudy_failures[first_index, second_index, third_index, line] * weight
            end
        end
    end
    return (zero_support, failed_weight > model.cloudy_touch_eps)
end

"""Reject invalid required fields instead of changing failed emission to zero."""
function require_nonnegative(value::Float64, name::String)
    isfinite(value) && value >= 0.0 || throw(DomainError(value, "$name must be finite and nonnegative"))
    return value
end

"""Calculate ten intrinsic line emissivities for one complete-z tile.

Input physical arrays have shape (Nx, Ny, Nz), in native Julia x-fastest order.
The output matrices are (Ncell, 10), with vec(tile.field) determining cell order.
Emissivities are erg/s/cm^3; each temperature [K] belongs to that line's gas
state. NaN denotes unavailable emission; prescribed cold CIII/CIV zeros remain
available. Dust is applied later using the model's stored cross-sections.
"""
function calculate_emission(model::EmissionModel, tile)
    density = vec(tile.nH_cm3)
    temperature_quokka = vec(tile.temperature_quokka_K)
    column = vec(tile.shielding_NH_cm2)
    gradient = vec(tile.dVdr_s)
    cell_count = length(density)
    for field in (temperature_quokka, column, gradient)
        length(field) == cell_count || throw(DimensionMismatch("Tile fields must have equal lengths"))
        all(isfinite, field) && all(field .> 0.0) || error("Emission inputs must be finite and positive")
    end
    emissivity = fill(NaN, cell_count, 10)
    temperature = fill(NaN, cell_count, 10)
    despotic_temperature = fill(NaN, cell_count)
    cold_cells = BitVector(temperature_quokka .< model.branch_temperature_K)
    indices = model.line_indices
    co10_index = indices["co10"]
    co21_index = indices["co21"]
    cii_index = indices["cii"]
    halpha_index = indices["halpha"]
    hi21_index = indices["hi21"]
    cold_zero_indices = Tuple(indices[key] for key in ("ciii_977", "ciii_1907", "ciii_1909", "civ_1548", "civ_1551"))
    despotic_clipped = zeros(Int, 3)
    cloudy_column_clipped = zeros(Int, 2)
    missing_despotic_temperature = 0
    failed_cloudy_queries = 0

    for cell in eachindex(density)
        query_coordinates = (
            despotic_coordinate(density[cell], model.despotic_axes[1], model.despotic_boundary_rtol, "nH"),
            despotic_coordinate(column[cell], model.despotic_axes[2], model.despotic_boundary_rtol, "NH"),
            despotic_coordinate(gradient[cell], model.despotic_axes[3], model.despotic_boundary_rtol, "dVdr"),
        )
        brackets = ntuple(axis -> interpolation_bracket(model.despotic_log_axes[axis], log10(query_coordinates[axis])), 3)
        equilibrium_temperature = trilinear_value(model.despotic_values, brackets, 1)
        despotic_temperature[cell] = equilibrium_temperature
        valid_despotic = isfinite(equilibrium_temperature) && equilibrium_temperature > 0.0
        if valid_despotic
            despotic_clipped[1] += query_coordinates[1] != density[cell]
            despotic_clipped[2] += query_coordinates[2] != column[cell]
            despotic_clipped[3] += query_coordinates[3] != gradient[cell]
            query_density = query_coordinates[1]
            co10 = require_nonnegative(trilinear_value(model.despotic_values, brackets, 2), "CO(1-0) lumPerH")
            co21 = require_nonnegative(trilinear_value(model.despotic_values, brackets, 3), "CO(2-1) lumPerH")
            emissivity[cell, co10_index] = query_density * co10
            emissivity[cell, co21_index] = query_density * co21
            temperature[cell, co10_index] = equilibrium_temperature
            temperature[cell, co21_index] = equilibrium_temperature
            if cold_cells[cell]
                cii = require_nonnegative(trilinear_value(model.despotic_values, brackets, 4), "CII lumPerH")
                electron_density = require_nonnegative(query_density * trilinear_value(model.despotic_values, brackets, 5), "Electron density")
                proton_density = require_nonnegative(query_density * trilinear_value(model.despotic_values, brackets, 6), "Proton density")
                neutral_density = require_nonnegative(query_density * trilinear_value(model.despotic_values, brackets, 7), "Neutral hydrogen density")
                temperature_4 = max(equilibrium_temperature / 1.0e4, 1.0e-10)
                exponent = -0.942 - 0.031 * log(temperature_4)
                recombination = 1.17e-13 * temperature_4^exponent
                emissivity[cell, cii_index] = query_density * cii
                emissivity[cell, halpha_index] = model.halpha_photon_energy_erg * recombination * electron_density * proton_density
                emissivity[cell, hi21_index] = model.hi21_emission_per_atom * neutral_density
                temperature[cell, cii_index] = equilibrium_temperature
                temperature[cell, halpha_index] = equilibrium_temperature
                temperature[cell, hi21_index] = equilibrium_temperature
            end
        else
            missing_despotic_temperature += 1
        end

        if cold_cells[cell]
            for line in cold_zero_indices
                emissivity[cell, line] = 0.0
                temperature[cell, line] = temperature_quokka[cell]
            end
            continue
        end

        log_column = log10(column[cell])
        log_density = cloudy_coordinate(log10(density[cell]), model.cloudy_log_axes[2], model.cloudy_boundary_rtol, "log_nH")
        log_temperature = cloudy_coordinate(log10(temperature_quokka[cell]), model.cloudy_log_axes[3], model.cloudy_boundary_rtol, "log_T")
        clipped_log_column = clamp(log_column, first(model.cloudy_log_axes[1]), last(model.cloudy_log_axes[1]))
        cloudy_coordinates = (clipped_log_column, log_density, log_temperature)
        cloudy_brackets = ntuple(axis -> interpolation_bracket(model.cloudy_log_axes[axis], cloudy_coordinates[axis]), 3)
        support = ntuple(line -> cloudy_support(model, cloudy_brackets, line), 8)
        if any(state -> state[2], support)
            # The adopted Cloudy failure rule excludes all eight atomic lines
            # together; independently available CO emission remains untouched.
            failed_cloudy_queries += 1
            continue
        end
        cloudy_column_clipped[1] += column[cell] < 10.0^first(model.cloudy_log_axes[1])
        cloudy_column_clipped[2] += column[cell] > 10.0^last(model.cloudy_log_axes[1])
        density_squared = density[cell]^2
        for line in 1:8
            coefficient = if support[line][1]
                trilinear_value(model.cloudy_values, cloudy_brackets, line)
            else
                10.0^trilinear_value(model.cloudy_log_values, cloudy_brackets, line)
            end
            emissivity[cell, line] = require_nonnegative(coefficient * density_squared, "Cloudy emissivity")
            temperature[cell, line] = temperature_quokka[cell]
        end
    end
    for value in emissivity
        isnan(value) || require_nonnegative(value, "Volume emissivity")
    end
    diagnostics = Dict{String,Any}(
        "despotic_temperature_missing_cells" => missing_despotic_temperature,
        "cloudy_failed_queries" => failed_cloudy_queries,
        "despotic_coordinate_clipped_cells" => Dict(zip(("nH", "NH", "dVdr"), despotic_clipped)),
        "cloudy_column_clipped_cells" => Dict(zip(("below", "above"), cloudy_column_clipped)),
        "line_missing_cells" => Dict(model.meta["line_keys"][line] => count(isnan, view(emissivity, :, line)) for line in 1:10),
    )
    return (
        emissivity_erg_s_cm3=emissivity,
        temperature_K=temperature,
        despotic_temperature_K=despotic_temperature,
        cold_cells=cold_cells,
        diagnostics=diagnostics,
    )
end
