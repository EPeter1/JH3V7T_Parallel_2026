__kernel void get_rank(
    __global const int* input,
    __global int* rank,
    const int size)
{
    int id = get_global_id(0);

    if (id >= size) {
        return;
    }

    int count = 0;

    for (int i = 0; i < size; i++) {
        if (input[i] < input[id]) {
            count++;
        }
    }

    rank[id] = count;
}
