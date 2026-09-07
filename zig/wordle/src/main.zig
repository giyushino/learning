const std = @import("std");
const rl = @import("raylib");

const screen_width = 1000;
const screen_height = 1400;

const cols = 10;
const visible_rows = 20;

// top left is 0,0
const cell_size = 60;
const origin_x = (screen_width - cols * cell_size) / 2;
const origin_y = (screen_height - visible_rows * cell_size) / 2;

const bg_color = rl.Color.init(18, 18, 24, 255);
const grid_color = rl.Color.init(45, 45, 58, 255);
const border_color = rl.Color.init(120, 120, 140, 255);

pub fn main(init: std.process.Init) !void {
    // var prng = std.Random.DefaultPrng.init(67);
    // const random = prng.random();
    _ = init;

    rl.initWindow(screen_width, screen_height, "wordle");
    defer rl.closeWindow();

    rl.setTargetFPS(60);

    while (!rl.windowShouldClose()) {
        rl.beginDrawing();
        rl.drawText("hello", 202, 80, 20, rl.Color.red);
        defer rl.endDrawing();
        rl.clearBackground(bg_color);
    }
}
