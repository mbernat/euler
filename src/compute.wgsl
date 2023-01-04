@group(0) @binding(0)
var<storage, read_write> field: array<f32>;
@group(1) @binding(0)
var output: texture_storage_2d<rgba8unorm, write>;

struct Ids {
    @builtin(local_invocation_index) local_index: u32,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
}

fn get_id(global_id: vec3<u32>, num_groups: vec3<u32>, group_size: u32) -> u32 {
    let row = num_groups.x * 16u;
    return global_id.y * row + global_id.x;
}

@compute @workgroup_size(16, 16)
fn compute(ids: Ids) {
    let id = get_id(ids.global_id, ids.num_groups, 16u);
    field[id] = (field[id] + sin(f32(id) / 65536.0)) % 1.0;
}

@compute @workgroup_size(16, 16)
fn copy(ids: Ids) {
    let id = get_id(ids.global_id, ids.num_groups, 16u);
    let color = vec4(field[id], field[id], field[id], 1.0);
    textureStore(output, vec2<i32>(ids.global_id.xy), color);
}