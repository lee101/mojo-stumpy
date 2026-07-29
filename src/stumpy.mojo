"""Matrix-profile and distance-profile kernels exposed through a C ABI."""

from std.algorithm import sync_parallelize
from std.math import pow, sqrt
from std.sys.info import simd_width_of

comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime W = simd_width_of[DType.float64]()
comptime PARALLEL_DISTANCE_THRESHOLD = 16_384


def fp(address: Int) -> Ptr:
    return Ptr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


@always_inline
def dot(a: Ptr, b: Ptr, n: Int) -> Float64:
    var lanes = SIMD[DType.float64, W](0.0)
    var i = 0
    while i + W <= n:
        lanes += a.load[width=W](i) * b.load[width=W](i)
        i += W
    var total = lanes.reduce_add()
    while i < n:
        total += a[i] * b[i]
        i += 1
    return total


@always_inline
def powered_delta(a: Float64, b: Float64, p_norm: Float64) -> Float64:
    var delta = abs(a - b)
    if p_norm == 2.0:
        return delta * delta
    if p_norm == 1.0:
        return delta
    return pow(delta, p_norm)


@always_inline
def insert_topk(
    profiles: Ptr,
    indices: IPtr,
    row: Int,
    k: Int,
    distance: Float64,
    neighbor: Int,
):
    var base = row * k
    if distance >= profiles[base + k - 1]:
        return
    var position = k - 1
    while position > 0 and distance < profiles[base + position - 1]:
        profiles[base + position] = profiles[base + position - 1]
        indices[base + position] = indices[base + position - 1]
        position -= 1
    profiles[base + position] = distance
    indices[base + position] = Int64(neighbor)


@always_inline
def normalized_distance(
    product: Float64,
    i: Int,
    j: Int,
    m: Int,
    means_a: Ptr,
    stds_a: Ptr,
    means_b: Ptr,
    stds_b: Ptr,
    constants_a: IPtr,
    constants_b: IPtr,
) -> Float64:
    if constants_a[i] != 0 and constants_b[j] != 0:
        return 0.0
    if constants_a[i] != 0 or constants_b[j] != 0:
        return sqrt(Float64(m))
    var denominator = Float64(m) * stds_a[i] * stds_b[j]
    var correlation = (product - Float64(m) * means_a[i] * means_b[j]) / denominator
    correlation = min(1.0, max(-1.0, correlation))
    var squared = 2.0 * Float64(m) * (1.0 - correlation)
    if squared < 1.0e-14:
        return 0.0
    return sqrt(squared)


@export("mst_profile")
def mst_profile(
    a_address: Int,
    b_address: Int,
    n_a: Int,
    n_b: Int,
    m: Int,
    means_a_address: Int,
    stds_a_address: Int,
    means_b_address: Int,
    stds_b_address: Int,
    valid_a_address: Int,
    valid_b_address: Int,
    constants_a_address: Int,
    constants_b_address: Int,
    k: Int,
    self_join: Int,
    normalized: Int,
    p_norm: Float64,
    workers: Int,
    profiles_address: Int,
    indices_address: Int,
    left_profiles_address: Int,
    right_profiles_address: Int,
    left_indices_address: Int,
    right_indices_address: Int,
) abi("C"):
    var a = fp(a_address)
    var b = fp(b_address)
    var means_a = fp(means_a_address)
    var stds_a = fp(stds_a_address)
    var means_b = fp(means_b_address)
    var stds_b = fp(stds_b_address)
    var valid_a = ip(valid_a_address)
    var valid_b = ip(valid_b_address)
    var constants_a = ip(constants_a_address)
    var constants_b = ip(constants_b_address)
    var profiles = fp(profiles_address)
    var indices = ip(indices_address)
    var left_profiles = fp(left_profiles_address)
    var right_profiles = fp(right_profiles_address)
    var left_indices = ip(left_indices_address)
    var right_indices = ip(right_indices_address)
    var length_a = n_a - m + 1
    var length_b = n_b - m + 1
    var exclusion = (m + 3) // 4
    var first_diagonal = -(length_a - 1)
    var diagonal_count = length_a + length_b - 1
    if self_join != 0:
        first_diagonal = exclusion + 1
        diagonal_count = max(0, length_a - first_diagonal)

    @parameter
    def process(worker: Int):
        var worker_profiles = profiles + worker * length_a * k
        var worker_indices = indices + worker * length_a * k
        var worker_left_profiles = left_profiles + worker * length_a
        var worker_right_profiles = right_profiles + worker * length_a
        var worker_left_indices = left_indices + worker * length_a
        var worker_right_indices = right_indices + worker * length_a
        var diagonal_index = worker
        while diagonal_index < diagonal_count:
            var diagonal = first_diagonal + diagonal_index
            var i = max(0, -diagonal)
            var j = i + diagonal
            var stop = min(length_a, length_b - diagonal)
            var accumulator = 0.0
            if i < stop:
                if normalized != 0:
                    accumulator = dot(a + i, b + j, m)
                else:
                    for offset in range(m):
                        accumulator += powered_delta(a[i + offset], b[j + offset], p_norm)
            while i < stop:
                if i > max(0, -diagonal):
                    if normalized != 0:
                        accumulator += (
                            a[i + m - 1] * b[j + m - 1]
                            - a[i - 1] * b[j - 1]
                        )
                    else:
                        accumulator += (
                            powered_delta(a[i + m - 1], b[j + m - 1], p_norm)
                            - powered_delta(a[i - 1], b[j - 1], p_norm)
                        )
                if valid_a[i] != 0 and valid_b[j] != 0:
                    var distance = 0.0
                    if normalized != 0:
                        distance = normalized_distance(
                            accumulator,
                            i,
                            j,
                            m,
                            means_a,
                            stds_a,
                            means_b,
                            stds_b,
                            constants_a,
                            constants_b,
                        )
                    else:
                        if p_norm == 2.0:
                            distance = sqrt(max(0.0, accumulator))
                        elif p_norm == 1.0:
                            distance = max(0.0, accumulator)
                        else:
                            distance = pow(max(0.0, accumulator), 1.0 / p_norm)
                    insert_topk(worker_profiles, worker_indices, i, k, distance, j)
                    if self_join != 0:
                        insert_topk(worker_profiles, worker_indices, j, k, distance, i)
                        if distance < worker_left_profiles[j]:
                            worker_left_profiles[j] = distance
                            worker_left_indices[j] = Int64(i)
                        if distance < worker_right_profiles[i]:
                            worker_right_profiles[i] = distance
                            worker_right_indices[i] = Int64(j)
                i += 1
                j += 1
            diagonal_index += workers

    sync_parallelize[process](workers)

    for worker in range(1, workers):
        var worker_profiles = profiles + worker * length_a * k
        var worker_indices = indices + worker * length_a * k
        for row in range(length_a):
            for rank in range(k):
                insert_topk(
                    profiles,
                    indices,
                    row,
                    k,
                    worker_profiles[row * k + rank],
                    Int(worker_indices[row * k + rank]),
                )
            var position = worker * length_a + row
            if left_profiles[position] < left_profiles[row]:
                left_profiles[row] = left_profiles[position]
                left_indices[row] = left_indices[position]
            if right_profiles[position] < right_profiles[row]:
                right_profiles[row] = right_profiles[position]
                right_indices[row] = right_indices[position]


@export("mst_distance_profile")
def mst_distance_profile(
    query_address: Int,
    series_address: Int,
    m: Int,
    n: Int,
    means_address: Int,
    stds_address: Int,
    valid_address: Int,
    constants_address: Int,
    query_mean: Float64,
    query_std: Float64,
    query_constant: Int,
    normalized: Int,
    p_norm: Float64,
    result_address: Int,
) abi("C"):
    var query = fp(query_address)
    var series = fp(series_address)
    var means = fp(means_address)
    var stds = fp(stds_address)
    var valid = ip(valid_address)
    var constants = ip(constants_address)
    var result = fp(result_address)
    var length = n - m + 1

    @parameter
    def compute(i: Int):
        if valid[i] == 0:
            return
        if normalized != 0:
            if query_constant != 0 and constants[i] != 0:
                result[i] = 0.0
            elif query_constant != 0 or constants[i] != 0:
                result[i] = sqrt(Float64(m))
            else:
                var product = dot(query, series + i, m)
                var denominator = Float64(m) * query_std * stds[i]
                var correlation = (
                    product - Float64(m) * query_mean * means[i]
                ) / denominator
                correlation = min(1.0, max(-1.0, correlation))
                var squared = 2.0 * Float64(m) * (1.0 - correlation)
                result[i] = 0.0 if squared < 1.0e-14 else sqrt(squared)
        else:
            var total = 0.0
            for offset in range(m):
                total += powered_delta(query[offset], series[i + offset], p_norm)
            if p_norm == 2.0:
                result[i] = sqrt(total)
            elif p_norm == 1.0:
                result[i] = total
            else:
                result[i] = pow(total, 1.0 / p_norm)

    if length >= PARALLEL_DISTANCE_THRESHOLD:
        sync_parallelize[compute](length)
    else:
        for i in range(length):
            compute(i)


@export("mst_normalize_products")
def mst_normalize_products(
    products_address: Int,
    means_address: Int,
    stds_address: Int,
    length: Int,
    m: Int,
    query_mean: Float64,
    query_std: Float64,
    result_address: Int,
) abi("C"):
    var products = fp(products_address)
    var means = fp(means_address)
    var stds = fp(stds_address)
    var result = fp(result_address)

    @parameter
    def compute_chunk(chunk: Int):
        comptime CHUNK_SIZE = 4_096
        var start = chunk * CHUNK_SIZE
        var stop = min(length, start + CHUNK_SIZE)
        var scale = Float64(m) * query_std
        var i = start
        while i + W <= stop:
            var correlations = (
                products.load[width=W](i)
                - Float64(m) * query_mean * means.load[width=W](i)
            ) / (scale * stds.load[width=W](i))
            correlations = min(
                SIMD[DType.float64, W](1.0),
                max(SIMD[DType.float64, W](-1.0), correlations),
            )
            var squared = (
                SIMD[DType.float64, W](2.0 * Float64(m))
                * (SIMD[DType.float64, W](1.0) - correlations)
            )
            squared = squared.lt(
                SIMD[DType.float64, W](1.0e-14)
            ).select(SIMD[DType.float64, W](0.0), squared)
            result.store(
                i, sqrt(max(SIMD[DType.float64, W](0.0), squared))
            )
            i += W
        while i < stop:
            var correlation = (
                products[i] - Float64(m) * query_mean * means[i]
            ) / (scale * stds[i])
            correlation = min(1.0, max(-1.0, correlation))
            var squared = 2.0 * Float64(m) * (1.0 - correlation)
            result[i] = 0.0 if squared < 1.0e-14 else sqrt(squared)
            i += 1

    comptime CHUNK_SIZE = 4_096
    var chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
    if length >= PARALLEL_DISTANCE_THRESHOLD:
        sync_parallelize[compute_chunk](chunks)
    else:
        for chunk in range(chunks):
            compute_chunk(chunk)
