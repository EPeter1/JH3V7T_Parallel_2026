__kernel void add_vectors(
    __global const float2* vector_1,
    __global const float2* vector_2,
    __global float2* sum,
    const int size)
{
    int id = get_global_id(0);

    if (id < size) {
        sum[id].x = vector_1[id].x + vector_2[id].x;
        sum[id].y = vector_1[id].y + vector_2[id].y;
    }
}
