# Stream level-0 native z layers without holding the snapshot's five fluid fields.

"""Require non-overlapping native z layers, covering every original z cell."""
function _stream_layer_ranges(intervals, nz::Int)
    ordered = sort!(unique!(collect(intervals)); by=first)
    isempty(ordered) && error("The snapshot has no native boxes")
    next_start = 1
    layers = UnitRange{Int}[]
    for (first_z, last_z) in ordered
        first_z == next_start || error(
            "Streaming requires native boxes aligned to non-overlapping z layers; " *
            "found an overlapping or incomplete interval $first_z:$last_z. " *
            "This layout needs a general box-cache reader before full-snapshot processing.",
        )
        first_z <= last_z <= nz || error("Native z-layer bounds are invalid")
        push!(layers, first_z:last_z)
        next_start = last_z + 1
    end
    next_start == nz + 1 || error("Native z layers do not cover the original domain")
    return layers
end


"""Budget the arrays retained by streaming; Mera temporaries and runtime are separate."""
function _stream_cache_plan(snapshot, layers; x_range, y_range, batch_cells, max_cache_bytes)
    plane_cells = length(x_range) * length(y_range)
    batch_cells >= plane_cells || error(
        "batch_cells=$batch_cells is smaller than one selected XY plane ($plane_cells cells); " *
        "increase batch_cells or select a smaller XY region",
    )
    max_cache_bytes > 0 || error("max_cache_bytes must be positive")
    batch_depth = max(1, batch_cells ÷ plane_cells)
    layer_depths = length.(layers)
    maximum_layer_depth = maximum(layer_depths)
    adjacent_depth = length(layers) == 1 ? layer_depths[1] : maximum(
        layer_depths[index] + layer_depths[index + 1] for index in 1:(length(layers) - 1)
    )
    halo_plane_cells = (length(x_range) + 2) * (length(y_range) + 2)
    memory_parts = (
        two_layer_fields_bytes=8 * 5 * halo_plane_cells * adjacent_depth,
        column_boundaries_bytes=8 * 2 * plane_cells * length(layers),
        current_reverse_column_bytes=8 * plane_cells * maximum_layer_depth,
        previous_velocity_face_bytes=8 * halo_plane_cells,
        column_workspace_bytes=8 * 3 * plane_cells,
        output_batch_bytes=8 * 7 * plane_cells * min(batch_depth, maximum_layer_depth),
        layer_fill_mask_bytes=halo_plane_cells * maximum_layer_depth,
    )
    retained_bytes = sum(values(memory_parts))
    retained_bytes <= max_cache_bytes || error(
        "Streaming needs an estimated $retained_bytes bytes of retained reader arrays, " *
        "exceeding max_cache_bytes=$max_cache_bytes. Increase the reader cache budget " *
        "after checking available memory, or select a smaller XY region/change the layout. " *
        "This estimate excludes Mera temporary buffers, the Julia runtime and emission/products.",
    )
    return (
        layers=layers,
        batch_depth=batch_depth,
        planned_retained_cache_bytes=retained_bytes,
        max_cache_bytes=max_cache_bytes,
        memory_parts=memory_parts,
    )
end


"""Plan z-layer reads and exact requested native-component bytes from FAB headers."""
function _stream_plan(snapshot; x_range, y_range, batch_cells, max_cache_bytes)
    header = Mera.read_amrex_header(snapshot.path)
    header.finest_level == 0 || error("Snapshot changed or contains unsupported AMR levels")
    level = only(header.levels)
    domain = level.prob_domain
    intervals = [(box.lo[3] - domain.lo[3] + 1, box.hi[3] - domain.lo[3] + 1) for box in level.boxes]
    layers = _stream_layer_ranges(intervals, snapshot.original_shape[3])
    plan = _stream_cache_plan(
        snapshot,
        layers;
        x_range=x_range,
        y_range=y_range,
        batch_cells=batch_cells,
        max_cache_bytes=max_cache_bytes,
    )
    nx, ny, _ = snapshot.original_shape
    requested_x = mod1.(collect((first(x_range) - 1):(last(x_range) + 1)), nx)
    requested_y = mod1.(collect((first(y_range) - 1):(last(y_range) + 1)), ny)
    low_x, high_x = extrema(requested_x)
    low_y, high_y = extrema(requested_y)
    component_bytes = 0
    boxes_to_visit = 0
    for (index, box) in enumerate(level.boxes)
        first_x = box.lo[1] - domain.lo[1] + 1
        last_x = box.hi[1] - domain.lo[1] + 1
        first_y = box.lo[2] - domain.lo[2] + 1
        last_y = box.hi[2] - domain.lo[2] + 1
        (last_x < low_x || first_x > high_x || last_y < low_y || first_y > high_y) && continue
        # The pinned Mera FAB-header parser reads no cell payload. Its precision
        # and ghost-inclusive dimensions give actual bytes for one component.
        fab_header = open(level.files[index], "r") do source
            Mera._read_amrex_fab_header(source, level.offsets[index], header.ndim)
        end
        fab_cells = prod(fab_header.box.hi[axis] - fab_header.box.lo[axis] + 1 for axis in 1:3)
        component_bytes += fab_cells * fab_header.bpr
        boxes_to_visit += 1
    end
    return merge(plan, (
        one_component_native_bytes=component_bytes,
        boxes_to_visit=boxes_to_visit,
    ))
end


"""Send stage/cell progress without keeping field arrays in the observer."""
function _stream_progress(progress; stage, completed_layers, total_layers, completed_cells, total_cells, z_range=nothing)
    progress === nothing && return nothing
    progress((
        stage=stage,
        completed_layers=completed_layers,
        total_layers=total_layers,
        completed_cells=completed_cells,
        total_cells=total_cells,
        z_range=z_range,
    ))
    return nothing
end


"""Precompute independent columns outside each layer, without total-minus-prefix subtraction."""
function _stream_column_boundaries(snapshot, plan; x_range, y_range, physics, progress)
    shape = (length(x_range), length(y_range), length(plan.layers))
    lower_columns = zeros(Float64, shape)
    upper_columns = zeros(Float64, shape)
    hydrogen_mass_g = Float64(physics["hydrogen_mass_g"])
    hydrogen_fraction = Float64(physics["X_H"])
    dz_cm = snapshot.cell_width_cm[3]
    boxes_visited = 0
    box_cell_visits = 0
    plane_cells = shape[1] * shape[2]
    total_cells = plane_cells * snapshot.original_shape[3]
    _stream_progress(progress; stage="density_prepass", completed_layers=0,
        total_layers=length(plan.layers), completed_cells=0, total_cells=total_cells)
    for (layer_index, z_range) in enumerate(plan.layers)
        fields = _reader_native_halo(
            snapshot;
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            density_only=true,
            announce_window=layer_index == 1,
        )
        boxes_visited += fields.reader_stats.boxes_visited
        box_cell_visits += fields.reader_stats.box_cell_visits
        for k in 1:length(z_range), j in 1:shape[2], i in 1:shape[1]
            hydrogen_density = fields.density[i + 1, j + 1, k] * hydrogen_fraction / hydrogen_mass_g
            lower_columns[i, j, layer_index] += hydrogen_density * dz_cm
        end
        _stream_progress(progress; stage="density_prepass", completed_layers=layer_index,
            total_layers=length(plan.layers), completed_cells=plane_cells * last(z_range),
            total_cells=total_cells, z_range=z_range)
    end
    all(isfinite, lower_columns) || error("Density prepass produced nonfinite layer columns")

    # Before overwriting the layer contributions with low-face columns, build
    # high-face columns in descending z order from those positive contributions.
    accumulated = zeros(Float64, shape[1], shape[2])
    for layer_index in length(plan.layers):-1:1, j in 1:shape[2], i in 1:shape[1]
        upper_columns[i, j, layer_index] = accumulated[i, j]
        accumulated[i, j] += lower_columns[i, j, layer_index]
    end
    fill!(accumulated, 0.0)
    for layer_index in 1:length(plan.layers), j in 1:shape[2], i in 1:shape[1]
        layer_column = lower_columns[i, j, layer_index]
        lower_columns[i, j, layer_index] = accumulated[i, j]
        accumulated[i, j] += layer_column
    end
    return (
        lower=lower_columns,
        upper=upper_columns,
        boxes_visited=boxes_visited,
        box_cell_visits=box_cell_visits,
    )
end


"""Emit bounded thin-z batches from one layer and its original velocity neighbours."""
function _stream_process_layer(
    callback,
    snapshot,
    current,
    following,
    previous_velocity_face,
    boundaries,
    plan;
    layer_index,
    x_range,
    y_range,
    physics,
    progress,
)
    z_range = plan.layers[layer_index]
    nx = length(x_range)
    ny = length(y_range)
    depth = length(z_range)
    hydrogen_mass_g = Float64(physics["hydrogen_mass_g"])
    hydrogen_fraction = Float64(physics["X_H"])
    gradient_floor_s = Float64(physics["dvdr_floor_s"])
    dx_cm, dy_cm, dz_cm = snapshot.cell_width_cm
    reverse_column = Array{Float64}(undef, nx, ny, depth)
    accumulated = copy(@view boundaries.upper[:, :, layer_index])
    for k in depth:-1:1, j in 1:ny, i in 1:nx
        hydrogen_density = current.density[i + 1, j + 1, k] * hydrogen_fraction / hydrogen_mass_g
        accumulated[i, j] += hydrogen_density * dz_cm
        reverse_column[i, j, k] = accumulated[i, j]
    end
    forward_column = copy(@view boundaries.lower[:, :, layer_index])
    total_cells = nx * ny * snapshot.original_shape[3]
    for first_local_z in 1:plan.batch_depth:depth
        last_local_z = min(first_local_z + plan.batch_depth - 1, depth)
        shape = (nx, ny, last_local_z - first_local_z + 1)
        density_g_cm3 = Array{Float64}(undef, shape)
        nH_cm3 = Array{Float64}(undef, shape)
        temperature_quokka_K = Array{Float64}(undef, shape)
        shielding_NH_cm2 = Array{Float64}(undef, shape)
        foreground_NH_cm2 = Array{Float64}(undef, shape)
        dVdr_s = Array{Float64}(undef, shape)
        velocity_z_kms = Array{Float64}(undef, shape)
        for k in first_local_z:last_local_z
            destination_k = k - first_local_z + 1
            global_k = first(z_range) + k - 1
            for j in 1:ny, i in 1:nx
                hi = i + 1
                hj = j + 1
                rho = current.density[hi, hj, k]
                hydrogen_density = rho * hydrogen_fraction / hydrogen_mass_g
                cell_column = hydrogen_density * dz_cm
                forward_column[i, j] += cell_column
                low_column = forward_column[i, j]
                high_column = reverse_column[i, j, k]
                density_g_cm3[i, j, destination_k] = rho
                nH_cm3[i, j, destination_k] = hydrogen_density
                temperature_quokka_K[i, j, destination_k] = current.temperature[hi, hj, k]
                velocity_z_kms[i, j, destination_k] = current.velocity_z[hi, hj, k] / 1.0e5
                shielding_NH_cm2[i, j, destination_k] = 2.0 / (1.0 / low_column + 1.0 / high_column)
                foreground_NH_cm2[i, j, destination_k] = low_column - 0.5 * cell_column

                dvx_dx = (current.velocity_x[hi + 1, hj, k] - current.velocity_x[hi - 1, hj, k]) / (2.0 * dx_cm)
                dvy_dy = (current.velocity_y[hi, hj + 1, k] - current.velocity_y[hi, hj - 1, k]) / (2.0 * dy_cm)
                central_velocity = current.velocity_z[hi, hj, k]
                previous_velocity = k > 1 ? current.velocity_z[hi, hj, k - 1] :
                    (global_k == 1 ? central_velocity : previous_velocity_face[hi, hj])
                next_velocity = k < depth ? current.velocity_z[hi, hj, k + 1] :
                    (global_k == snapshot.original_shape[3] ? central_velocity : following.velocity_z[hi, hj, 1])
                z_denominator = global_k == 1 || global_k == snapshot.original_shape[3] ? dz_cm : 2.0 * dz_cm
                dvz_dz = (next_velocity - previous_velocity) / z_denominator
                dVdr_s[i, j, destination_k] = max(abs(dvx_dx + dvy_dy + dvz_dz) / 3.0, gradient_floor_s)
            end
        end
        all(value -> isfinite(value) && value > 0, nH_cm3) || error("Derived hydrogen density is invalid")
        all(value -> isfinite(value) && value > 0, shielding_NH_cm2) || error("Derived shielding column is invalid")
        all(value -> isfinite(value) && value >= 0, foreground_NH_cm2) || error("Derived foreground column is invalid")
        all(isfinite, dVdr_s) || error("Derived velocity gradient is invalid")
        batch_z_range = (first(z_range) + first_local_z - 1):(first(z_range) + last_local_z - 1)
        callback((
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
            z_range=batch_z_range,
        ))
        _stream_progress(progress; stage="processing",
            completed_layers=last_local_z == depth ? layer_index : layer_index - 1,
            total_layers=length(plan.layers), completed_cells=nx * ny * last(batch_z_range),
            total_cells=total_cells, z_range=batch_z_range)
    end
    return nothing
end


"""Execute a checked stream plan; callbacks must consume each batch before returning."""
function _stream_execute(callback, snapshot, plan; x_range, y_range, physics, progress)
    boundaries = _stream_column_boundaries(
        snapshot,
        plan;
        x_range=x_range,
        y_range=y_range,
        physics=physics,
        progress=progress,
    )
    main_boxes = 0
    main_cell_visits = 0
    function load_layer(index)
        fields = _reader_native_halo(
            snapshot;
            x_range=x_range,
            y_range=y_range,
            z_range=plan.layers[index],
            announce_window=false,
        )
        main_boxes += fields.reader_stats.boxes_visited
        main_cell_visits += fields.reader_stats.box_cell_visits
        return fields
    end
    current = load_layer(1)
    following = length(plan.layers) > 1 ? load_layer(2) : nothing
    previous_velocity_face = nothing
    for layer_index in eachindex(plan.layers)
        _stream_process_layer(
            callback,
            snapshot,
            current,
            following,
            previous_velocity_face,
            boundaries,
            plan;
            layer_index=layer_index,
            x_range=x_range,
            y_range=y_range,
            physics=physics,
            progress=progress,
        )
        if layer_index < length(plan.layers)
            previous_velocity_face = copy(@view current.velocity_z[:, :, end])
            current = following
            following = layer_index + 2 <= length(plan.layers) ? load_layer(layer_index + 2) : nothing
        end
    end
    boundaries.boxes_visited == plan.boxes_to_visit || error("Density-pass box visits differ from the fixed stream plan")
    main_boxes == plan.boxes_to_visit || error("Main-pass box visits differ from the fixed stream plan")
    return (
        density_pass=(
            boxes_visited=boundaries.boxes_visited,
            box_cell_visits=boundaries.box_cell_visits,
            native_payload_bytes=plan.one_component_native_bytes,
        ),
        main_pass=(
            boxes_visited=main_boxes,
            box_cell_visits=main_cell_visits,
            native_payload_bytes=5 * plan.one_component_native_bytes,
        ),
        total_layers=length(plan.layers),
        batch_depth=plan.batch_depth,
        planned_retained_cache_bytes=plan.planned_retained_cache_bytes,
        max_cache_bytes=plan.max_cache_bytes,
        memory_parts=plan.memory_parts,
        cache_budget_excludes="Mera temporary arrays, Julia runtime, emission/products, and arrays retained by callbacks",
    )
end


"""
    foreach_snapshot_batch(callback, snapshot; x_range, y_range, physics,
                           batch_cells=131072, max_cache_bytes=1024^3, progress=nothing)

Stream complete selected XY planes in ascending, bounded z batches. A density
prepass builds independent columns outside each native z layer. The main pass
reads each selected FAB once for five fields, retaining at most adjacent layers
and the previous velocity face. No full-snapshot fluid field is retained.

Native boxes must form non-overlapping z layers. Each batch has the seven reader
arrays in `(x,y,z)` order plus global 1-based `x_range`, `y_range`, `z_range`.
Callbacks must finish using the batch before returning and must not retain its
arrays. The cache budget covers the listed retained reader arrays, not process
RSS; Mera I/O/conversion temporaries, runtime and downstream work are additional.
"""
function foreach_snapshot_batch(
    callback,
    snapshot;
    x_range::UnitRange{Int}=1:snapshot.original_shape[1],
    y_range::UnitRange{Int}=1:snapshot.original_shape[2],
    physics::AbstractDict,
    batch_cells::Int=131_072,
    max_cache_bytes::Int=1024^3,
    progress=nothing,
)
    isempty(x_range) && error("The x range must not be empty")
    isempty(y_range) && error("The y range must not be empty")
    1 <= first(x_range) <= last(x_range) <= snapshot.original_shape[1] || error("x_range lies outside the native grid")
    1 <= first(y_range) <= last(y_range) <= snapshot.original_shape[2] || error("y_range lies outside the native grid")
    hydrogen_mass_g = Float64(physics["hydrogen_mass_g"])
    hydrogen_fraction = Float64(physics["X_H"])
    gradient_floor_s = Float64(physics["dvdr_floor_s"])
    isfinite(hydrogen_mass_g) && hydrogen_mass_g > 0 || error("hydrogen_mass_g must be finite and positive")
    isfinite(hydrogen_fraction) && 0 < hydrogen_fraction <= 1 || error("X_H must be in (0, 1]")
    isfinite(gradient_floor_s) && gradient_floor_s > 0 || error("dvdr_floor_s must be finite and positive")
    plan = _stream_plan(
        snapshot;
        x_range=x_range,
        y_range=y_range,
        batch_cells=batch_cells,
        max_cache_bytes=max_cache_bytes,
    )
    return _stream_execute(
        callback,
        snapshot,
        plan;
        x_range=x_range,
        y_range=y_range,
        physics=physics,
        progress=progress,
    )
end
