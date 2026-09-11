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

fn cellRect(coord: Coord) rl.Rectangle {
    return .{
        .x = @floatFromInt(origin_x + coord.col * cell_size),
        .y = @floatFromInt(origin_y + coord.row * cell_size),
        .width = cell_size,
        .height = cell_size,
    };
}

pub fn drawLetter(char: [:0]const u8, col: i32, row: i32) void {
    const rect = cellRect(Coord{ .col = col, .row = row });
    rl.drawRectangleLinesEx(rect, 2, rl.Color.red);

    // center the glyph inside the cell
    const text_width: f32 = @floatFromInt(rl.measureText(char, font_size));
    const text_x = rect.x + (rect.width - text_width) / 2;
    const text_y = rect.y + (rect.height - font_size) / 2;
    rl.drawText(char, @intFromFloat(text_x), @intFromFloat(text_y), font_size, rl.Color.white);
}

pub fn drawWord(x_off: i32, y_off: i32) void {
    const rect = cellRect(Coord{ .col = x_off, .row = y_off });
    rl.drawRectangleLinesEx(rect, 2, rl.Color.red);
}


fn countLines(text: []const u8) usize {
    @setEvalBranchQuota(1_000_000);
    var n: usize = 0;
    var iterator = std.mem.tokenizeScalar(u8, text, '\n'); 
    while (iterator.next()) |_| { n += 1; }
    return n;
}

pub fn createWordList(comptime words: []const u8) [countLines(words)][5]u8 {
    @setEvalBranchQuota(20_000);
    var out: [countLines(words)][5] u8 = undefined;
    var i: usize = 0;
    var it = std.mem.tokenizeScalar(u8, words, '\n');
    while (it.next()) |word| {
        out[i] = word[0..5].*;
        i+= 1;
    }
    return out;
}
 

pub fn oldMain(init: std.process.Init) !void {
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
        drawLetter("h", 0, 0);
        drawLetter("e", 1, 0);
    }
}


pub fn main(init: std.process.Init) !void {
    _ = init;

    const words = @embedFile("words.txt");
    const word_list: [countLines(words)][5]u8 = createWordList(words);
    std.debug.print("{s}", .{ word_list[0] });
}
