use macroquad::prelude::*;

struct Grid {
    width: usize,
    height: usize,
    u: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    s: Vec<Vec<f32>>,
    rho: Vec<Vec<f32>>,
    p: Vec<Vec<f32>>,
}

impl Grid {
    fn new(width: usize, height: usize) -> Grid {
        let zero_col = vec![0.0; height + 2];
        let one_col = vec![1.0; height + 2];

        let mut u = vec![zero_col.clone(); width + 2];
        let v = vec![zero_col.clone(); width + 2];
        let mut s = vec![one_col; width + 2];
        let mut rho = vec![zero_col.clone(); width + 2];
        let p = vec![zero_col; width + 2];

        // vertical walls
        for j in 0..height + 2 {
            s[0][j] = 0.0;
            s[width + 1][j] = 0.0;
        }

        // horizontal walls
        for i in 0..width + 2 {
            s[i][0] = 0.0;
            s[i][height + 1] = 0.0;
        }

        let radius = 15.0;
        let pos = vec2(width as f32 / 5.0, height as f32 / 2.0);
        for i in 1..=width {
            for j in 1..=height {
                if vec2(i as f32, j as f32).distance_squared(pos) < radius * radius {
                    s[i][j] = 0.0;
                    rho[i][j] = 1.0;
                }
            }
        }

        u[1][height / 2] = 100.0;

        Grid {
            width,
            height,
            u,
            v,
            s,   // 1.0 fluid, 0.0 solid
            rho, // density
            p,   // pressure
        }
    }

    fn step(&mut self, dt: f32) {
        // Gravity
        let g = -9.81;

        for i in 1..=self.width {
            for j in 1..=self.height {
                if self.s[i][j] != 0.0 && self.s[i][j - 1] != 0.0 {
                    self.v[i][j] += g * dt;
                }
            }
        }

        for i in 1..=self.width {
            for j in 1..=self.height {
                self.p[i][j] = 0.0;
            }
        }

        let rho = 1000.0;
        let size = 0.1;
        let over_relaxation = 1.9;

        for _iter in 0..100 {
            // Projection
            for i in 1..=self.width {
                for j in 1..=self.height {
                    let s = &self.s;
                    let ss = s[i - 1][j] + s[i][j - 1] + s[i + 1][j] + s[i][j + 1];
                    if ss == 0.0 {
                        continue;
                    }
                    let d = over_relaxation * {
                        let u = &self.u;
                        let v = &self.v;
                        u[i + 1][j] - u[i][j] + v[i][j + 1] - v[i][j]
                    };
                    self.u[i][j] += d * s[i - 1][j] / ss;
                    self.u[i + 1][j] -= d * s[i + 1][j] / ss;
                    self.v[i][j] += d * s[i][j - 1] / ss;
                    self.v[i][j + 1] -= d * s[i][j + 1] / ss;
                    self.p[i][j] += d / ss * rho * size / dt;
                }
            }
        }

        let pu = self.u.clone();
        let pv = self.v.clone();

        // Advect u; indices i = 0, 1 width + 1 are in/on the wall
        for i in 2..=self.width {
            for j in 1..=self.height {
                if self.s[i][j] != 0.0 && self.s[i - 1][j] != 0.0 {
                    let v = (pv[i - 1][j] + pv[i][j] + pv[i - 1][j + 1] + pv[i][j + 1]) / 4.0;
                    // We align these vectors to the u grid
                    let x = vec2(i as f32, j as f32);
                    let vel = vec2(pu[i][j], v);
                    // The real grid is `size` times bigger than the integral grid.
                    let p = x - vel * dt / size;
                    self.u[i][j] = self.sample_field(&pu, p);
                }
            }
        }

        // Advect v; indices j = 0, 1, height + 1 are in/on the wall
        for i in 1..=self.width {
            for j in 2..=self.height {
                if self.s[i][j] != 0.0 && self.s[i][j - 1] != 0.0 {
                    let u = (pu[i][j - 1] + pu[i + 1][j - 1] + pu[i][j] + pu[i + 1][j]) / 4.0;
                    let vel = vec2(u, pv[i][j]);
                    let x = vec2(i as f32, j as f32);
                    let p = x - vel * dt / size;
                    self.v[i][j] = self.sample_field(&pv, p);
                }
            }
        }

        let pr = self.rho.clone();

        // Advect density
        for i in 1..=self.width {
            for j in 1..=self.height {
                if self.s[i][j] != 0.0 {
                    let u = (self.u[i][j] + self.u[i + 1][j]) / 2.0;
                    let v = (self.v[i][j] + self.v[i][j + 1]) / 2.0;
                    let vel = vec2(u, v);
                    let x = vec2(i as f32, j as f32);
                    let p = x - vel * dt / size;
                    self.rho[i][j] = self.sample_field(&pr, p);
                }
            }
        }
    }

    fn sample_field(&self, field: &Vec<Vec<f32>>, p: Vec2) -> f32 {
        let pi = p.x.floor();
        let pj = p.y.floor();
        let x = p.x - pi;
        let y = p.y - pj;
        let pi = (pi as usize).clamp(1, self.width + 1);
        let pj = (pj as usize).clamp(1, self.height + 1);
        let res = field[pi][pj] * (1.0 - x) * (1.0 - y)
            + field[pi + 1][pj] * x * (1.0 - y)
            + field[pi][pj + 1] * (1.0 - x) * y
            + field[pi + 1][pj + 1] * x * y;
        res
    }

    fn render(&self) {
        let pixels = 5.0;
        let start = vec2(0.0 - pixels, 500.0 + pixels);

        for i in 1..=self.width {
            for j in 1..=self.height {
                let u = (self.u[i][j] + self.u[i + 1][j]) / 2.0;
                let v = (self.v[i][j] + self.v[i][j + 1]) / 2.0;
                let pos = start + vec2(i as f32 * pixels, 0.0) - vec2(0.0, j as f32 * pixels);

                // Velocity
                let speed = vec2(u, v).length();
                let norm = 1.0;
                //                let color = [u.abs() / norm, v.abs() / norm, speed / norm, 1.0].into();
                let color: Color = [u / norm, -u / norm, 0.0, 1.0].into();

                // Pressure
                let color: Color = [self.p[i][j] / 100000.0, 0.0, 0.0, 1.0].into();

                // Density
                let rho = self.rho[i][j];
                let color = [rho, 0.0, 0.0, 1.0].into();

                draw_rectangle(pos.x, pos.y, pixels, pixels, color);
            }
        }

        for i in 1..=self.width {
            for j in 1..=self.height {
                let u = (self.u[i][j] + self.u[i + 1][j]) / 2.0;
                let v = (self.v[i][j] + self.v[i][j + 1]) / 2.0;
                let pos = start + vec2(i as f32 * pixels, 0.0) - vec2(0.0, j as f32 * pixels);
                let pos = pos + vec2(pixels / 2.0, -pixels / 2.0);
                let d = vec2(u, v) * 2.0;
                let color: Color = [0.0, 0.0, 1.0, 0.3].into();
                draw_line(pos.x, pos.y, pos.x + d.x, pos.y - d.y, 1.0, color);
            }
        }
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Euler".to_string(),
        window_width: 1000,
        window_height: 500,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut grid = Grid::new(200, 100);

    let mut accu = 0.0;
    let time = 0.01;

    loop {
        let dt = get_frame_time();
        accu += dt;
        grid.step(dt);
        if accu > time {
            accu -= time;
            //     grid.step(0.01);
        }
        grid.render();

        next_frame().await;
    }
}
