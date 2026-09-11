const std = @import("std");
// Rendering lives in src/root.zig (the "wordle" module). Re-enable when hooking
// the UI back up.
// const render = @import("wordle");

const Results = enum { correct, partial, invalid };


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


pub fn checkWords(guess: []const u8, sol: []const u8, valid_words: []const [5]u8 ) Results {
    if (std.mem.eql(u8, guess, sol)) return Results.correct;

    var l: usize = 0; var r: usize = 0;
    while (l < r) {
        const mid: usize = l + ((r - l) / 2);
        const word: [5]u8  = valid_words[mid];

        if (std.mem.eql(u8, guess, &word)) return Results.correct;

        const compare = std.mem.order(u8, guess, &word);
        switch (compare) {
            .lt => l = mid,
            .eq => return Results.partial,
            .gt => r = mid,
        }
    }

    return Results.invalid;
}


pub fn main(init: std.process.Init) !void {
    _ = init;

    const words = @embedFile("words.txt");
    const word_list: [countLines(words)][5]u8 = createWordList(words);
    std.debug.print("{s}", .{ word_list[0] });
}
