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

const Piece = enum { i, j, l, o, s, t, z };
const box_size = [7]usize{ 4, 3, 3, 2, 3, 3, 3 };
const piece_colors = [7]rl.Color{
    .init(0, 240, 240, 255),   // I cyan
    .init(0, 0, 240, 255),     // J blue
    .init(240, 160, 0, 255),   // L orange
    .init(240, 240, 0, 255),   // O yellow
    .init(0, 240, 0, 255),     // S green
    .init(160, 0, 240, 255),   // T purple
    .init(240, 0, 0, 255),     // Z red
};
const Mask = u16;
const Coord = struct { col: i32, row: i32 };




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
        var piece = spawn;
        shapes[idx][0] = maskFromGrid(piece);
        // overwrite piece each time
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

fn cellRect(coord: Coord) rl.Rectangle {
    return .{
        .x = @floatFromInt(origin_x + coord.col * cell_size),
        .y = @floatFromInt(origin_y + coord.row * cell_size),
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

fn cells(piece: Piece, rotation: u2, x_off: i32, y_off: i32) [4]Coord {
    var occupied_cells: [4]Coord = undefined;
    var num_found: u4 = 0; // as most 4 pieces
    var piece_mask = SHAPES[@intFromEnum(piece)][rotation];
    
    for (0..16) |i| {
        if (piece_mask & 1 != 0) {
            occupied_cells[num_found] = Coord{
                .col = x_off + @as(i32, @intCast(i % 4)),
                .row = y_off + @as(i32, @intCast(i / 4))
            };
            num_found += 1;
        }
        piece_mask >>= 1;
    }
    return occupied_cells;
}

fn drawShape(piece:Piece, rotation: u2, x_off: i32, y_off: i32) void {
    const color = piece_colors[@intFromEnum(piece)];
    const occupied_cells = cells(piece, rotation, x_off, y_off);

    for (occupied_cells) |cell| {
        var rect = cellRect(cell);
        // later we should just move this into
        // cell rect
        rect.width -= 2;
        rect.height -= 2;
        rl.drawRectangleRec(rect, color);
    }
}


// for now just want to see if we can draw to screen a rectangle
fn testDrawSquare() void {
    rl.drawRectangle(0, 0, cell_size, cell_size, rl.Color.red);
    const coords = Coord{
        .col = 0,
        .row = 10
    };
    rl.drawRectangleRec(cellRect(coords), rl.Color.red);
}

pub fn main(init: std.process.Init) !void {
    _ = init;

    rl.initWindow(screen_width, screen_height, "tetris");
    defer rl.closeWindow();

    rl.setTargetFPS(60);
    
    const update_y_pos_rate: u8 = std.math.maxInt(u8);
    var rot: u2 = 0;
    var frames: u32 = 0;
    var x_pos: i32 = 4;
    var y_pos: i32 = 0;

    while (!rl.windowShouldClose()) {
        frames += 1;
        rl.beginDrawing();
        defer rl.endDrawing();
        
        rl.clearBackground(bg_color);
        drawGrid();
        testDrawSquare();
        
        if (rl.isKeyPressed(.right)) {
            x_pos = @min(x_pos + 1, cols);
        }
        if (rl.isKeyPressed(.left)) {
            x_pos = @max(x_pos - 1, 0);
        }

        if (rl.isKeyPressed(.up)) {
            rot -%= 1;
        }
        drawShape(Piece.i, rot, x_pos, y_pos);
        if (frames % update_y_pos_rate == 0) {
            y_pos += 1;
        }
    }
}
