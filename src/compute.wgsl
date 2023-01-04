struct Point {
    u: f32,
    v: f32,
    s: f32,
    rho: f32,
    avg_div: f32,
    nu: f32,
    nv: f32,
    nrho: f32,
}

@group(0) @binding(0)
var<storage, read_write> field: array<Point>;
@group(1) @binding(0)
var output: texture_storage_2d<rgba8unorm, write>;

struct Ids {
    @builtin(local_invocation_index) local_index: u32,
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(num_workgroups) num_groups: vec3<u32>,
}

fn get_id(global_id: vec3<u32>, num_groups: vec3<u32>, group_size: u32) -> u32 {
    let row = num_groups.x * group_size;
    return global_id.y * row + global_id.x;
}

@compute @workgroup_size(16, 16)
fn integrate(ids: Ids) {
    // TODO pass dt & g as constants
    let dt = 0.01;
    let g = -9.81;
    // TODO pass size as a constant
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    if field[index].s == 1.0 && field[index - row_size].s == 1.0 {
        field[index].v += g * dt;
    }
}

@compute @workgroup_size(16, 16)
fn gather(ids: Ids) {
    // TODO pass size as a constant
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    //let index = get_id(ids.global_id, ids.num_groups, 1u);
    let here = field[index];
    if here.s == 1.0 {
        let l = index - 1u;
        let r = index + 1u;
        let b = index - row_size;
        let t = index + row_size;
        let left = field[l];
        let right = field[r];
        let bottom = field[b];
        let top = field[t];
        let s = left.s + right.s + bottom.s + top.s;
        let avg_div = (right.u - here.u + top.v - here.v) / s;
        field[index].avg_div = avg_div;
    }
}

@compute @workgroup_size(16, 16)
fn scatter_bottom_left(ids: Ids) {
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    if field[index].s == 1.0 {
        let l = index - 1u;
        let b = index - row_size;
        let left = field[l];
        let bottom = field[b];
        let avg_div = field[index].avg_div;
        field[index].u += left.s * avg_div;
        field[index].v += bottom.s * avg_div;
    }
}

@compute @workgroup_size(16, 16)
fn scatter_top_right(ids: Ids) {
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    if field[index].s == 1.0 {
        let r = index + 1u;
        let t = index + row_size;
        let right = field[r];
        let top = field[t];
        let avg_div = field[index].avg_div;
        field[r].u -= right.s * avg_div;
        field[t].v -= top.s * avg_div;
    }
}

@compute @workgroup_size(16, 16)
fn advect_u(ids: Ids) {
    let dt = 0.01;
    let cell_size = 0.1;
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    let here = field[index];
    if here.s == 1.0 && field[index - 1u].s == 1.0 {
        let u = here.u;
        let v = (here.v + field[index - 1u].v + field[index + row_size].v + field[index + row_size - 1u].v) / 4.0;
        // TODO try abstracting field sampling
        let p = vec2<f32>(ids.global_id.xy) - vec2(u, v) * dt / cell_size;
        let p_id = vec2<u32>(floor(p));
        let b = p - floor(p);
        let a = 1.0 - b;
        let p_index = p_id.y * row_size + p_id.x;
        let nu =
              field[p_index].u * a.x * a.y
            + field[p_index + 1u].u * b.x * a.y
            + field[p_index + row_size].u * a.x * b.y
            + field[p_index + row_size + 1u].u * b.x * b.y;
        field[index].nu = nu;
    }
}

@compute @workgroup_size(16, 16)
fn advect_v(ids: Ids) {
    let dt = 0.01;
    let cell_size = 0.1;
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    let here = field[index];
    if here.s == 1.0 && field[index - row_size].s == 1.0 {
        let u = (here.u + field[index - row_size].u + field[index + 1u].u + field[index - row_size + 1u].u) / 4.0;
        let v = here.v;
        let p = vec2<f32>(ids.global_id.xy) - vec2(u, v) * dt / cell_size;
        let p_id = vec2<u32>(floor(p));
        let b = p - floor(p);
        let a = 1.0 - b;
        let p_index = p_id.y * row_size + p_id.x;
        let nv =
              field[p_index].v * a.x * a.y
            + field[p_index + 1u].v * b.x * a.y
            + field[p_index + row_size].v * a.x * b.y
            + field[p_index + row_size + 1u].v * b.x * b.y;
        field[index].nv = nv;
    }
}

@compute @workgroup_size(16, 16)
fn advect_density(ids: Ids) {
    let dt = 0.01;
    let cell_size = 0.1;
    let row_size = 256u;
    let id = ids.global_id;
    let index = id.y * row_size + id.x;
    let here = field[index];
    field[index].nrho = here.rho;
    if here.s == 1.0 {
        let u = (here.u + field[index + 1u].u) / 2.0;
        let v = (here.v + field[index + row_size].v) / 2.0;
        let p = vec2<f32>(ids.global_id.xy) - vec2(u, v) * dt / cell_size;
        let p_id = vec2<u32>(floor(p));
        let b = p - floor(p);
        let a = 1.0 - b;
        let p_index = p_id.y * row_size + p_id.x;
        let nrho =
              field[p_index].rho * a.x * a.y
            + field[p_index + 1u].rho * b.x * a.y
            + field[p_index + row_size].rho * a.x * b.y
            + field[p_index + row_size + 1u].rho * b.x * b.y;
        field[index].nrho = nrho;
    }
}

@compute @workgroup_size(16, 16)
fn copy(ids: Ids) {
    let id = get_id(ids.global_id, ids.num_groups, 16u);
    let f = field[id];
    field[id].u = f.nu;
    field[id].v = f.nv;
    // Copy velocity before advecting density
    field[id].rho = f.nrho;
    let f = field[id];
    let speed = length(vec2(f.u, f.v));
    //let color = vec4(abs(f.u) / 10.0, speed / 100.0, (f.v + 5.0) / 10.0, f.s);
    let color = vec4(f.rho, f.rho, f.rho, 1.0);
    textureStore(output, vec2<i32>(ids.global_id.xy), color);
}