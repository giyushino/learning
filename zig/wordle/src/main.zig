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


pub fn checkWords(guess: []const u8, sol: []const u8, valid_words: []const u8 ) Results {
    if (std.mem.eql(u8, guess, sol)) return Results.correct;

    var l: usize = 0; var r: usize = 0;
    while (l < r) {
        const mid: usize = l + ((r - l) / 2);
        const w: [5]u8  = valid_words[mid * stride ..][0..5].*;

        if (std.mem.eql(u8, guess, &w)) return Results.correct;

        const compare = std.mem.order(u8, guess, &w);
        switch (compare) {
            .lt => l = mid + 1,
            .eq => return Results.partial,
            .gt => r = mid,
        }
    }

    return Results.invalid;
}


pub fn main(init: std.process.Init) !void {
    _ = init;

    std.debug.print("{s}\n", .{ word(0) });
    const result: Results = checkWords("hello", "tests", words);
    switch (result) {
        .correct => std.debug.print("correct\n", .{}),
        .partial => std.debug.print("partial\n", .{}),
        .invalid => std.debug.print("invalid\n", .{}),
    }
}
