# Read uniform QUOKKA plotfiles into bounded, complete-z numerical tiles.

import Mera


"""
    open_snapshot(path)

Read geometry and units without loading the fluid cell table. The first Julia
implementation accepts 3-D, Cartesian, uniform level-0 plotfiles with stored
temperature and explicit physical unit factors in `metadata.yaml`.
"""
function open_snapshot(path::AbstractString)
    plotfile_path = realpath(expanduser(path))
    output_match = match(r"^plt(\d+)$", basename(plotfile_path))
    output_match === nothing && error("Expected a snapshot directory named plt followed by digits: $plotfile_path")
    output_number = parse(Int, output_match.captures[1])
    header = Mera.read_amrex_header(plotfile_path)
    header.ndim == 3 || error("Julia process requires a three-dimensional snapshot")
    header.coord == 0 || error("Julia process requires Cartesian snapshot coordinates")
    header.finest_level == 0 || error("Julia process currently supports uniform level-0 snapshots, not AMR")

    required_fields = ("gasDensity", "x-GasMomentum", "y-GasMomentum", "z-GasMomentum")
    for field_name in required_fields
        field_name in header.fields || error("Required stored QUOKKA field is missing: $field_name")
    end
    has_stored_temperature = "temperature" in header.fields || "gasTemperature" in header.fields
    has_stored_temperature || error("A stored temperature field is required; temperature reconstruction is not enabled")

    # Mera permits dimensionless fallback units. Physical emission processing
    # instead requires explicit positive factors, including temperature.
    metadata = Mera.read_quokka_metadata(plotfile_path)
    haskey(metadata, "units") || error("metadata.yaml must declare physical units")
    metadata_units = metadata["units"]
    for unit_name in ("unit_length", "unit_mass", "unit_time", "unit_temperature")
        haskey(metadata_units, unit_name) || error("metadata.yaml is missing $unit_name")
        unit_value = Float64(metadata_units[unit_name])
        isfinite(unit_value) && unit_value > 0 || error("$unit_name must be finite and positive")
    end

    info = Mera.getinfo_quokka(
        output_number,
        plotfile_path;
        verbose=false,
    )
    domain_indices = header.prob_domain[1]
    original_shape = ntuple(axis -> domain_indices.hi[axis] - domain_indices.lo[axis] + 1, 3)
    all(size -> size > 0, original_shape) || error("Snapshot dimensions must be positive")
    original_shape[3] >= 2 || error("At least two z cells are required for the physical boundary gradient")
    cell_width_cm = ntuple(axis -> Float64(header.dx[1][axis] * info.unit_l), 3)
    all(width -> isfinite(width) && width > 0, cell_width_cm) || error("Cell widths must be finite and positive")
    domain_left_cm = ntuple(axis -> Float64(header.domain_lo[axis] * info.unit_l), 3)
    domain_right_cm = ntuple(axis -> Float64(header.domain_hi[axis] * info.unit_l), 3)
    all(isfinite, domain_left_cm) || error("Domain left edges must be finite")
    all(isfinite, domain_right_cm) || error("Domain right edges must be finite")
    for axis in 1:3
        expected_width = original_shape[axis] * cell_width_cm[axis]
        actual_width = domain_right_cm[axis] - domain_left_cm[axis]
        isapprox(actual_width, expected_width; rtol=1e-10, atol=0.0) || error("Header geometry is inconsistent on axis $axis")
    end
    density_unit_g_cm3 = Float64(info.unit_d)
    velocity_unit_cm_s = Float64(info.unit_v)
    isfinite(density_unit_g_cm3) && density_unit_g_cm3 > 0 || error("Density unit must be finite and positive")
    isfinite(velocity_unit_cm_s) && velocity_unit_cm_s > 0 || error("Velocity unit must be finite and positive")
    cell_volume_cm3 = prod(cell_width_cm)
    isfinite(cell_volume_cm3) && cell_volume_cm3 > 0 || error("Cell volume must be finite and positive")
    time_s = Float64(header.time * info.unit_t)
    isfinite(time_s) || error("Snapshot time must be finite")

    return (
        path=plotfile_path,
        info=info,
        original_shape=original_shape,
        cell_width_cm=cell_width_cm,
        cell_volume_cm3=cell_volume_cm3,
        domain_left_cm=domain_left_cm,
        domain_right_cm=domain_right_cm,
        density_unit_g_cm3=density_unit_g_cm3,
        velocity_unit_cm_s=velocity_unit_cm_s,
        time_s=time_s,
        levelmin=header.levelmin,
    )
end


"""Map each requested original index to its destinations, including periodic copies."""
function _reader_periodic_positions(indices::Vector{Int})
    positions = Dict{Int,Vector{Int}}()
    for (destination, original_index) in enumerate(indices)
        push!(get!(positions, original_index, Int[]), destination)
    end
    return positions
end


"""Read a native z interval and periodic x/y halo, optionally density alone."""
function _reader_native_halo(
    snapshot;
    x_range::UnitRange{Int},
    y_range::UnitRange{Int},
    z_range::UnitRange{Int}=1:snapshot.original_shape[3],
    density_only::Bool=false,
    announce_window::Bool=true,
)
    nx, ny, nz = snapshot.original_shape
    x_indices = mod1.(collect((first(x_range) - 1):(last(x_range) + 1)), nx)
    y_indices = mod1.(collect((first(y_range) - 1):(last(y_range) + 1)), ny)
    x_positions = _reader_periodic_positions(x_indices)
    y_positions = _reader_periodic_positions(y_indices)
    halo_shape = (length(x_indices), length(y_indices), length(z_range))
    density = Array{Float64}(undef, halo_shape)
    temperature = density_only ? nothing : Array{Float64}(undef, halo_shape)
    velocity_x = density_only ? nothing : Array{Float64}(undef, halo_shape)
    velocity_y = density_only ? nothing : Array{Float64}(undef, halo_shape)
    velocity_z = density_only ? nothing : Array{Float64}(undef, halo_shape)
    filled = falses(halo_shape)

    # Mera windows use its enclosing cube, not each physical domain's length.
    # A wrapped halo can span both faces. One encompassing scan reads both,
    # while the destination maps retain only the requested cells.
    lattice_size = 2.0^snapshot.levelmin
    window_x = [(minimum(x_indices) - 1) / lattice_size, maximum(x_indices) / lattice_size]
    window_y = [(minimum(y_indices) - 1) / lattice_size, maximum(y_indices) / lattice_size]
    window_z = [(first(z_range) - 1) / lattice_size, last(z_range) / lattice_size]
    envelope_xy_cells = (maximum(x_indices) - minimum(x_indices) + 1) * (maximum(y_indices) - minimum(y_indices) + 1)
    requested_unique_xy_cells = length(x_positions) * length(y_positions)
    if announce_window && envelope_xy_cells > requested_unique_xy_cells
        @info "Periodic halo requires an encompassing Mera window; unrequested cells are discarded" envelope_xy_cells requested_unique_xy_cells
    end

    box_cell_visits = 0
    selected_variables = density_only ? [:rho] : [:rho, :temperature, :vx, :vy, :vz]
    boxes_visited = Mera.amrex_foreach_box(
        snapshot.info;
        vars=selected_variables,
        xrange=window_x,
        yrange=window_y,
        zrange=window_z,
        max_threads=1,
        verbose=false,
    ) do chunk
        chunk.level == 0 || error("Unexpected AMR chunk while reading a uniform snapshot")
        box_cell_visits += prod(chunk.dims)
        origin = Mera.amrex_box_origin(chunk)
        source_x = [i for i in 1:chunk.dims[1] if haskey(x_positions, origin[1] + i)]
        source_y = [j for j in 1:chunk.dims[2] if haskey(y_positions, origin[2] + j)]
        (isempty(source_x) || isempty(source_y)) && return nothing
        for k in 1:chunk.dims[3]
            original_z = origin[3] + k
            1 <= original_z <= nz || error("A native box extends outside the original z domain")
            original_z in z_range || continue
            destination_z = original_z - first(z_range) + 1
            for j in source_y
                for i in source_x
                    chunk.mask[i, j, k] || continue
                    source_index = i + chunk.dims[1] * (j - 1 + chunk.dims[2] * (k - 1))
                    rho_cgs = chunk.cols.rho[source_index] * snapshot.density_unit_g_cm3
                    if !density_only
                        temperature_K = chunk.cols.temperature[source_index]
                        vx_cgs = chunk.cols.vx[source_index] * snapshot.velocity_unit_cm_s
                        vy_cgs = chunk.cols.vy[source_index] * snapshot.velocity_unit_cm_s
                        vz_cgs = chunk.cols.vz[source_index] * snapshot.velocity_unit_cm_s
                    end
                    for destination_y in y_positions[origin[2] + j]
                        for destination_x in x_positions[origin[1] + i]
                            filled[destination_x, destination_y, destination_z] && error("Overlapping level-0 boxes cover a requested cell twice")
                            density[destination_x, destination_y, destination_z] = rho_cgs
                            if !density_only
                                temperature[destination_x, destination_y, destination_z] = temperature_K
                                velocity_x[destination_x, destination_y, destination_z] = vx_cgs
                                velocity_y[destination_x, destination_y, destination_z] = vy_cgs
                                velocity_z[destination_x, destination_y, destination_z] = vz_cgs
                            end
                            filled[destination_x, destination_y, destination_z] = true
                        end
                    end
                end
            end
        end
    end
    all(filled) || error("Native boxes did not fill every requested tile and halo cell")
    all(value -> isfinite(value) && value > 0, density) || error("Tile and halo density must be finite and positive")
    if !density_only
        all(value -> isfinite(value) && value > 0, temperature) || error("Tile and halo temperature must be finite and positive")
        all(isfinite, velocity_x) || error("Tile and halo x velocity must be finite")
        all(isfinite, velocity_y) || error("Tile and halo y velocity must be finite")
        all(isfinite, velocity_z) || error("Tile and halo z velocity must be finite")
    end

    return (
        density=density,
        temperature=temperature,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z,
        reader_stats=(
            boxes_visited=boxes_visited,
            box_cell_visits=box_cell_visits,
            unique_requested_cells=requested_unique_xy_cells * length(z_range),
            window_cells=envelope_xy_cells * length(z_range),
        ),
    )
end


"""
    read_tile(snapshot; x_range, y_range, physics)

Return owned Float64 `(length(x_range), length(y_range), Nz)` arrays in Julia
native x-fastest order. Ranges are 1-based original-grid indices; z is complete.
`physics` supplies `hydrogen_mass_g`, `X_H` and `dvdr_floor_s` from the Python
physics export. Shielding includes each full cell in both directions; the
foreground toward the outer -z observer includes half the emitting cell.
"""
function read_tile(snapshot; x_range::UnitRange{Int}, y_range::UnitRange{Int}, physics::AbstractDict)
    isempty(x_range) && error("The x range must not be empty")
    isempty(y_range) && error("The y range must not be empty")
    1 <= first(x_range) <= last(x_range) <= snapshot.original_shape[1] || error("x_range lies outside the native grid")
    1 <= first(y_range) <= last(y_range) <= snapshot.original_shape[2] || error("y_range lies outside the native grid")
    hydrogen_mass_g = Float64(physics["hydrogen_mass_g"])
    hydrogen_mass_fraction = Float64(physics["X_H"])
    gradient_floor_s = Float64(physics["dvdr_floor_s"])
    isfinite(hydrogen_mass_g) && hydrogen_mass_g > 0 || error("hydrogen_mass_g must be finite and positive")
    isfinite(hydrogen_mass_fraction) && 0 < hydrogen_mass_fraction <= 1 || error("X_H must be in (0, 1]")
    isfinite(gradient_floor_s) && gradient_floor_s > 0 || error("dvdr_floor_s must be finite and positive")
    fields = _reader_native_halo(snapshot; x_range=x_range, y_range=y_range)
    tile_shape = (length(x_range), length(y_range), snapshot.original_shape[3])
    density_g_cm3 = Array{Float64}(undef, tile_shape)
    nH_cm3 = Array{Float64}(undef, tile_shape)
    temperature_quokka_K = Array{Float64}(undef, tile_shape)
    shielding_NH_cm2 = Array{Float64}(undef, tile_shape)
    foreground_NH_cm2 = Array{Float64}(undef, tile_shape)
    dVdr_s = Array{Float64}(undef, tile_shape)
    velocity_z_kms = Array{Float64}(undef, tile_shape)
    dx_cm, dy_cm, dz_cm = snapshot.cell_width_cm
    nz = tile_shape[3]

    # Core fields and velocity divergence use original physical neighbours.
    # Only the two original z faces use one-sided differences.
    for k in 1:nz
        for j in 1:tile_shape[2]
            for i in 1:tile_shape[1]
                hi = i + 1
                hj = j + 1
                rho = fields.density[hi, hj, k]
                density_g_cm3[i, j, k] = rho
                nH_cm3[i, j, k] = rho * hydrogen_mass_fraction / hydrogen_mass_g
                temperature_quokka_K[i, j, k] = fields.temperature[hi, hj, k]
                velocity_z_kms[i, j, k] = fields.velocity_z[hi, hj, k] / 1.0e5
                dvx_dx = (fields.velocity_x[hi + 1, hj, k] - fields.velocity_x[hi - 1, hj, k]) / (2.0 * dx_cm)
                dvy_dy = (fields.velocity_y[hi, hj + 1, k] - fields.velocity_y[hi, hj - 1, k]) / (2.0 * dy_cm)
                if k == 1
                    dvz_dz = (fields.velocity_z[hi, hj, 2] - fields.velocity_z[hi, hj, 1]) / dz_cm
                elseif k == nz
                    dvz_dz = (fields.velocity_z[hi, hj, nz] - fields.velocity_z[hi, hj, nz - 1]) / dz_cm
                else
                    dvz_dz = (fields.velocity_z[hi, hj, k + 1] - fields.velocity_z[hi, hj, k - 1]) / (2.0 * dz_cm)
                end
                divergence_s = dvx_dx + dvy_dy + dvz_dz
                dVdr_s[i, j, k] = max(abs(divergence_s) / 3.0, gradient_floor_s)
            end
        end
    end

    # Accumulate from each boundary independently. This avoids subtracting
    # nearly equal total and foreground columns across large density contrasts.
    plus_column_cm2 = zeros(Float64, tile_shape[1], tile_shape[2])
    for k in nz:-1:1
        for j in 1:tile_shape[2]
            for i in 1:tile_shape[1]
                plus_column_cm2[i, j] += nH_cm3[i, j, k] * dz_cm
                shielding_NH_cm2[i, j, k] = plus_column_cm2[i, j]
            end
        end
    end
    minus_column_cm2 = zeros(Float64, tile_shape[1], tile_shape[2])
    cumulative_nH_cm3 = zeros(Float64, tile_shape[1], tile_shape[2])
    for k in 1:nz
        for j in 1:tile_shape[2]
            for i in 1:tile_shape[1]
                hydrogen_density = nH_cm3[i, j, k]
                minus_column_cm2[i, j] += hydrogen_density * dz_cm
                cumulative_nH_cm3[i, j] += hydrogen_density
                plus_column = shielding_NH_cm2[i, j, k]
                minus_column = minus_column_cm2[i, j]
                shielding_NH_cm2[i, j, k] = 2.0 / (1.0 / plus_column + 1.0 / minus_column)
                foreground_NH_cm2[i, j, k] = (cumulative_nH_cm3[i, j] - 0.5 * hydrogen_density) * dz_cm
            end
        end
    end
    all(value -> isfinite(value) && value > 0, nH_cm3) || error("Derived hydrogen density is invalid")
    all(value -> isfinite(value) && value > 0, shielding_NH_cm2) || error("Derived shielding column is invalid")
    all(value -> isfinite(value) && value >= 0, foreground_NH_cm2) || error("Derived foreground column is invalid")
    all(isfinite, dVdr_s) || error("Derived velocity gradient is invalid")

    return (
        nH_cm3=nH_cm3,
        temperature_quokka_K=temperature_quokka_K,
        shielding_NH_cm2=shielding_NH_cm2,
        foreground_NH_cm2=foreground_NH_cm2,
        dVdr_s=dVdr_s,
        velocity_z_kms=velocity_z_kms,
        density_g_cm3=density_g_cm3,
        cell_width_cm=snapshot.cell_width_cm,
        cell_volume_cm3=snapshot.cell_volume_cm3,
        domain_left_cm=snapshot.domain_left_cm,
        original_shape=snapshot.original_shape,
        x_range=x_range,
        y_range=y_range,
        reader_stats=fields.reader_stats,
    )
end
