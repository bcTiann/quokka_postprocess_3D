"""Persistent numerical products for an ordered sequence of z-plane batches.

Only one native x-y image and small task-owned raw spectra/moments survive a
batch. Each batch first accumulates bounded raw partials, then merges them into
the persistent state to limit long-chain floating-point summation error. No
tile, emissivity matrix, or channel-cell matrix is retained. Task shards own
fixed, disjoint x-y columns, independent of physical thread scheduling.
"""
mutable struct ProductAccumulator{G,P,S}
    geometry::G
    parameters::P
    images::Array{Float64,4}
    states::Vector{S}
    column_ranges::Vector{UnitRange{Int}}
    next_z::Int
    processed_cells::Int
    planned_cells::Int
    batch_count::Int
    finished::Bool
    failed::Bool
end

"""
    ProductAccumulator(geometry, meta; threads=Threads.nthreads())

Create one accumulator for the planned native x/y/z ranges. `geometry`
provides 1-based inclusive `x_range`, `y_range`, and `z_range`, physical
`cell_width_cm` (three values), and scalar `cell_volume_cm3`. Typically z is
`1:Nz`, even when x/y select a region. `meta` is the same fixed-physics JSON
dictionary accepted by `accumulate_products`.

Every submitted batch covers the complete selected x/y plane and a contiguous
part of the planned z range. The caller owns snapshot reads and complete-column
physics. With a 256-by-256 image and ten lines, the shared images occupy 10 MiB;
spectra/moments/scratch scale with the requested task count, not snapshot depth.
"""
function ProductAccumulator(geometry, meta; threads=Threads.nthreads())
    parameters = product_parameters(meta)
    for range in (geometry.x_range, geometry.y_range, geometry.z_range)
        range isa AbstractUnitRange{<:Integer} || throw(ArgumentError("Product ranges must be integer unit ranges"))
        !isempty(range) && first(range) >= 1 || throw(ArgumentError("Product ranges must be nonempty and 1-based"))
    end
    length(geometry.cell_width_cm) == 3 || throw(ArgumentError("Expected three physical cell widths"))
    all(isfinite, geometry.cell_width_cm) && all(geometry.cell_width_cm .> 0.0) ||
        throw(ArgumentError("Product cell widths must be finite and positive"))
    isfinite(geometry.cell_volume_cm3) && geometry.cell_volume_cm3 > 0.0 ||
        throw(ArgumentError("Product cell volume must be finite and positive"))
    1 <= threads <= Threads.nthreads() || throw(ArgumentError("Invalid number of product tasks"))
    # Keep only small value metadata, never a reader or array-owning tile.
    plan = (
        x_range=Int(first(geometry.x_range)):Int(last(geometry.x_range)),
        y_range=Int(first(geometry.y_range)):Int(last(geometry.y_range)),
        z_range=Int(first(geometry.z_range)):Int(last(geometry.z_range)),
        cell_width_cm=Tuple(Float64.(geometry.cell_width_cm)),
        cell_volume_cm3=Float64(geometry.cell_volume_cm3),
    )
    nx = length(plan.x_range)
    ny = length(plan.y_range)
    column_count = Base.checked_mul(nx, ny)
    planned_cells = Base.checked_mul(column_count, length(plan.z_range))
    task_count = min(Int(threads), column_count)
    column_ranges = [
        fld((task - 1) * column_count, task_count) + 1:fld(task * column_count, task_count)
        for task in 1:task_count
    ]
    channel_count = length(parameters.edges) - 1
    line_count = length(parameters.masses_g)
    phase_count = length(parameters.phase_bounds) + 2
    states = [new_product_accumulator(channel_count, line_count, phase_count) for _ in 1:task_count]
    images = zeros(nx, ny, line_count, 2)
    return ProductAccumulator(
        plan,
        parameters,
        images,
        states,
        column_ranges,
        first(plan.z_range),
        0,
        planned_cells,
        0,
        false,
        false,
    )
end

"""Merge bounded raw batch partials after all task shards finish successfully.

Centroids and dispersions are never averaged here. The merge combines raw
weights, centered second moments, spectra and counts in a fixed task order.
This avoids accumulating tens of millions of cell increments into one state.
"""
function accumulate_product_batch_tasks!(accumulator, tile, emission)
    if length(accumulator.states) == 1
        partial = accumulate_product_columns!(
            accumulator.images,
            only(accumulator.column_ranges),
            tile,
            emission,
            accumulator.parameters,
        )
        merge_product_accumulators!(only(accumulator.states), partial)
        return nothing
    end
    tasks = [
        Threads.@spawn accumulate_product_columns!(
            accumulator.images,
            accumulator.column_ranges[index],
            tile,
            emission,
            accumulator.parameters,
        )
        for index in eachindex(accumulator.states)
    ]
    failure = nothing
    partials = similar(accumulator.states)
    for index in eachindex(tasks)
        try
            partials[index] = fetch(tasks[index])
        catch exception
            isnothing(failure) && (failure = exception)
        end
    end
    isnothing(failure) || throw(failure)
    for index in eachindex(partials)
        merge_product_accumulators!(accumulator.states[index], partials[index])
    end
    return nothing
end

"""
    add_batch!(accumulator, tile, emission)

Add one complete-x/y, partial-z numerical batch and release all input references
before returning. Tile arrays are `(Nx,Ny,Nz_batch)` with x fastest; emission
matrices use `(Nx*Ny*Nz_batch,line)` in the same cell order. `tile.z_range`
records the original 1-based indices, e.g. `65:66`, not local batch indices.

Batch ranges must cover the planned z interval once in increasing order. A
shape/range error is rejected before numerical mutation. A worker failure marks
the accumulator unusable: callers must discard it rather than retrying a batch
that may already have contributed. No partial batch is reported as processed.
"""
function add_batch!(accumulator::ProductAccumulator, tile, emission)
    accumulator.finished && error("Cannot add a batch after products were finalized")
    accumulator.failed && error("Cannot reuse product state after a failed numerical batch")
    geometry = accumulator.geometry
    tile.x_range == geometry.x_range || throw(DimensionMismatch("Batch x range changed"))
    tile.y_range == geometry.y_range || throw(DimensionMismatch("Batch y range changed"))
    tile.cell_width_cm == geometry.cell_width_cm || throw(ArgumentError("Batch cell widths changed"))
    tile.cell_volume_cm3 == geometry.cell_volume_cm3 || throw(ArgumentError("Batch cell volume changed"))
    tile.z_range isa AbstractUnitRange{<:Integer} || throw(ArgumentError("Batch z_range must be an integer unit range"))
    !isempty(tile.z_range) || throw(ArgumentError("Batch z_range must not be empty"))
    first(tile.z_range) == accumulator.next_z || error("Batch z range is out of order or overlaps a previous batch")
    last(tile.z_range) <= last(geometry.z_range) || error("Batch z range exceeds the planned domain")
    size(tile.density_g_cm3, 3) == length(tile.z_range) || throw(DimensionMismatch("Batch z range length"))
    check_product_batch(tile, emission, length(accumulator.parameters.masses_g))
    try
        accumulate_product_batch_tasks!(accumulator, tile, emission)
    catch
        accumulator.failed = true
        rethrow()
    end
    accumulator.next_z = last(tile.z_range) + 1
    accumulator.processed_cells += length(tile.density_g_cm3)
    accumulator.batch_count += 1
    return accumulator
end

"""
    finish_products(accumulator)

Merge raw task spectra and stable weighted moments once, then prepare all final
arrays and check cell/luminosity/mass conservation. Incomplete z coverage cannot
produce a completed product. Output keys and dimensions match
`accumulate_products`, including per-line missing counts/masses. Normalization,
centroids and dispersions are calculated only here, never per submitted batch.
"""
function finish_products(accumulator::ProductAccumulator)
    accumulator.finished && error("Products were already finalized")
    accumulator.failed && error("Cannot finalize products after a failed numerical batch")
    accumulator.next_z == last(accumulator.geometry.z_range) + 1 ||
        error("Cannot finalize products before the planned z range is complete")
    accumulator.processed_cells == accumulator.planned_cells || error("Processed cell count differs from the plan")
    parameters = accumulator.parameters
    combined = new_product_accumulator(
        length(parameters.edges) - 1,
        length(parameters.masses_g),
        length(parameters.phase_bounds) + 2,
    )
    for state in accumulator.states
        merge_product_accumulators!(combined, state)
    end
    only(combined.processed_cell_count) == accumulator.processed_cells ||
        error("Product workers counted a different number of cells")
    output = build_product_output(
        accumulator.images,
        combined,
        accumulator.geometry,
        parameters,
    )
    try
        check_product_accounting(output)
    catch
        accumulator.failed = true
        rethrow()
    end
    accumulator.finished = true
    return output
end
