//! Raylib rendering for wordle. No game rules here — see src/main.zig.
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
const font_size = 40;

const Coord = struct { col: i32, row: i32 };
const colors = [3]rl.Color{
    .init(197, 197, 255, 255), // purple
    .init(241,195,56, 255), // yellow
    .init(197, 255, 226, 255) // green
};

fn cellRect(coord: Coord) rl.Rectangle {
    return .{
        .x = @floatFromInt(origin_x + coord.col * cell_size),
        .y = @floatFromInt(origin_y + coord.row * cell_size),
        .width = cell_size,
        .height = cell_size,
    };
}

pub fn drawLetter(char: u8, col: i32, row: i32, color: rl.Color) void {
    const rect = cellRect(Coord{ .col = col, .row = row });

    // raylib wants a null-terminated C string, so give the byte a home
    const buf: [1:0]u8 = .{char};
    const text: [:0]const u8 = &buf;
    rl.drawRectangleRec(rect, color);

    // center the glyph inside the cell
    const text_width: f32 = @floatFromInt(rl.measureText(text, font_size));
    const text_x = rect.x + (rect.width - text_width) / 2;
    const text_y = rect.y + (rect.height - font_size) / 2;
    rl.drawText(text, @intFromFloat(text_x), @intFromFloat(text_y), font_size, rl.Color.white);
}

pub fn drawWord(word: []const u8, row: i32, correctness: [5] u8) void {
    for (0.., word) |idx, char| {
        drawLetter(char, @intCast(idx), row, colors[correctness[idx]]);
    }
}


pub fn render(init: std.process.Init) !void {
    // var prng = std.Random.DefaultPrng.init(67);
    // const random = prng.random();
    _ = init;
    rl.initWindow(screen_width, screen_height, "wordle");
    defer rl.closeWindow();

    rl.setTargetFPS(60);

    while (!rl.windowShouldClose()) {
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(bg_color);
        drawLetter('h', 0, 0, colors[0]);
        drawLetter('e', 1, 0, colors[1]);
        drawWord("hello", 2, .{1, 0, 1, 1, 2});
    }
}
