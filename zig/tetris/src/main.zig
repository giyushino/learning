const std = @import("std");
const rl = @import("raylib");

const screen_width = 600;
const screen_height = 800;

const cols = 10;
const visible_rows = 20;

// top left is 0, 0
const cell_size = 32;
const origin_x = (screen_width - cols * cell_size) / 2;
const origin_y = (screen_height - visible_rows * cell_size) / 2;

const bg_color = rl.Color.init(18, 18, 24, 255);
const grid_color = rl.Color.init(45, 45, 58, 255);
const border_color = rl.Color.init(120, 120, 140, 255);

const Piece = enum { i, j, l, o, s, t, z };
const box_size = [7]usize{ 4, 3, 3, 2, 3, 3, 3 };
const Mask = u16;

fn genShapes() [7][4]Mask {
    // first explicity write out what the shapes
    // look like in a nested arr, then we can rotate,
    // and write all of this to binary
    const i_piece = [4][4]u8{
        .{0, 1, 0, 0},
        .{0, 1, 0, 0},
        .{0, 1, 0, 0},
        .{0, 1, 0, 0},
    };
    const j_piece = [4][4]u8{
        .{1, 0, 0, 0},
        .{1, 1, 1, 0},
        .{0, 0, 0, 0},
        .{0, 0, 0, 0},
    };
    const l_piece = [4][4]u8{
        .{0, 0, 1, 0},
        .{1, 1, 1, 0},
        .{0, 0, 0, 0},
        .{0, 0, 0, 0},
    };
    const o_piece = [4][4]u8{
        .{1, 1, 0, 0},
        .{1, 1, 0, 0},
        .{0, 0, 0, 0},
        .{0, 0, 0, 0},
    };
    const s_piece = [4][4]u8{
        .{0, 1, 1, 0},
        .{1, 1, 0, 0},
        .{0, 0, 0, 0},
        .{0, 0, 0, 0},
    };
    const t_piece = [4][4]u8{
        .{0, 1, 0, 0},
        .{1, 1, 1, 0},
        .{0, 0, 0, 0},
        .{0, 0, 0, 0},
    };
    const z_piece = [4][4]u8{
        .{1, 1, 0, 0},
        .{0, 1, 1, 0},
        .{0, 0, 0, 0},
        .{0, 0, 0, 0},
    };

    // ordering must match the pieces enum
    const spawn_pieces = [7][4][4]u8{ i_piece, j_piece, l_piece, o_piece, s_piece, t_piece, z_piece };

    var shapes: [7][4]Mask = undefined;
    for (spawn_pieces, 0..) |spawn, idx| {
        // Each rotation is the previous one turned once more, so turns
        // accumulate instead of every rotation starting over from spawn.
        var piece = spawn;
        shapes[idx][0] = maskFromGrid(piece);
        for (1..4) |rot| {
            piece = rotatePiece(piece, box_size[idx]);
            shapes[idx][rot] = maskFromGrid(piece);
        }
    }
    return shapes;
}

// rotate counterclockwise
fn rotatePiece(piece: [4][4]u8, n: usize) [4][4]u8 {
    // we can't set this as undefined since we might
    // only rotate the top left n by n square
    var rotated_piece = std.mem.zeroes([4][4]u8);

    for (0..n) |row| {
        for (0..n) |col| {
            rotated_piece[row][col] = piece[col][n - 1 - row];
        }
    }
    return rotated_piece;
}

fn maskFromGrid(piece: [4][4]u8) Mask {
    var result: Mask = 0;

    for (piece, 0..) |row, r| {
        for (row, 0..) |cell, c| {
            if (cell != 0) {
                const bit: u4 = @intCast(r * 4 + c);
                result |= @as(Mask, 1) << bit;
            }
        }
    }
    return result;
}

const SHAPES = genShapes();

fn cellRect(col: i32, row: i32) rl.Rectangle {
    return .{
        .x = @floatFromInt(origin_x + col * cell_size),
        .y = @floatFromInt(origin_y + row * cell_size),
        .width = cell_size,
        .height = cell_size,
    };
}

// Draws the empty playfield: interior grid lines plus an outer border.
fn drawGrid() void {
    const left = origin_x;
    const top = origin_y;
    const right = origin_x + cols * cell_size;
    const bottom = origin_y + visible_rows * cell_size;

    // Interior lines only; the outer edges are covered by the border below.
    for (1..cols) |i| {
        const x: i32 = left + @as(i32, @intCast(i)) * cell_size;
        rl.drawLine(x, top, x, bottom, grid_color);
    }
    for (1..visible_rows) |i| {
        const y: i32 = top + @as(i32, @intCast(i)) * cell_size;
        rl.drawLine(left, y, right, y, grid_color);
    }

    rl.drawRectangleLinesEx(.{
        .x = @floatFromInt(left),
        .y = @floatFromInt(top),
        .width = cols * cell_size,
        .height = visible_rows * cell_size,
    }, 2, border_color);
}

fn drawShape(piece: Piece) void {
}

/// Prints all four rotations of a piece as ASCII, so transcription errors in
/// the shape table are visible at a glance.
fn printShape(piece: Piece) void {
    const rotations = SHAPES[@intFromEnum(piece)];
    std.debug.print("--- {s} ---\n", .{@tagName(piece)});
    for (rotations, 0..) |mask, rot| {
        std.debug.print("rotation {d}: {b:0>16}\n", .{ rot, mask });
        for (0..4) |r| {
            for (0..4) |c| {
                const bit: u4 = @intCast(r * 4 + c);
                const filled = mask & (@as(Mask, 1) << bit) != 0;
                std.debug.print("{s}", .{if (filled) "#" else "."});
            }
            std.debug.print("\n", .{});
        }
        std.debug.print("\n", .{});
    }
}

// for now just want to see if we can draw to screen a rectangle
fn testDrawSquare() void {
    rl.drawRectangle(0, 0, cell_size, cell_size, rl.Color.red);
    rl.drawRectangleRec(cellRect(10, 0), rl.Color.red);
}

pub fn main(init: std.process.Init) !void {
    _ = init;

    // printShape(.j);

    rl.initWindow(screen_width, screen_height, "tetris");
    defer rl.closeWindow();

    rl.setTargetFPS(60);

    // Main game loop: runs until the window is closed or Esc is pressed.
    while (!rl.windowShouldClose()) {
        rl.beginDrawing();
        defer rl.endDrawing();

        rl.clearBackground(bg_color);
        drawGrid();
        testDrawSquare();
    }
}
