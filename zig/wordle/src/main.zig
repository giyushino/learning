const std = @import("std");
// Rendering lives in src/root.zig (the "wordle" module). Re-enable when hooking
// the UI back up.
// const render = @import("wordle");

const Results = enum { correct, partial, invalid };

const words = @embedFile("words.txt");
const stride = 6;
const word_count = words.len / stride;
comptime { std.debug.assert(words.len % stride == 0); }

fn word(i: usize) *const [5]u8 {
    return words[i * stride ..][0..5];
}


pub fn checkGuess(guess: []const u8, sol: []const u8, valid_words: []const u8 ) Results {
    if (std.mem.eql(u8, guess, sol)) return Results.correct;

    var l: usize = 0; var r: usize = word_count;
    while (l < r) {
        const mid: usize = l + ((r - l) / 2);
        const w: [5]u8  = valid_words[mid * stride ..][0..5].*;

        const compare = std.mem.order(u8, guess, &w);
        switch (compare) {
            .lt => r = mid,
            .eq => return Results.partial,
            .gt => l = mid + 1,
        }
    }

    return Results.invalid;
}


pub fn scoreGuess(guess: []const u8, sol: []const u8, allocator: std.mem.Allocator) ![5]u8 {
    var correctness = [_]u8{0} ** 5 ;
    var map: std.AutoHashMap(u8, u4) = .init(allocator);
    defer map.deinit();

    for (sol) |char| {
        const count = map.get(char) orelse 0;
        try map.put(char, count + 1);
    }

    for (0.., sol, guess) |idx, s_char, g_char| {
        if (s_char == g_char) {
            correctness[idx] = 2;
            const count = map.get(s_char).?;
            try map.put(s_char, count - 1);
        }
    }

    for (0.., guess) |idx, char| {
        if (correctness[idx] ==  0) {
            const count = map.get(char) orelse 0;
            if (count != 0) { 
                correctness[idx] = 1;
                try map.put(char, count - 1);
            } 
        }
    }

    return correctness;
}


pub fn main(init: std.process.Init) !void {
    _ = init;

    var gpa: std.heap.DebugAllocator(.{}) = .init;
    const allocator = gpa.allocator();

    std.debug.print("{s}\n", .{ word(0) });
    const result: Results = checkGuess("hello", "tests", words);
    switch (result) {
        .correct => std.debug.print("correct\n", .{}),
        .partial => std.debug.print("partial\n", .{}),
        .invalid => std.debug.print("invalid\n", .{}),
    }
    const correctness = scoreGuess("cccch", "cockc", allocator);
    _ = try correctness; 
}



