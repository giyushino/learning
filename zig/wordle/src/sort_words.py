def sort_words(file_path: str):
    words = []
    with open(file_path, "r") as file:
        for line in file:
            words.append(line)

    print(words[:2])
    words.sort()
    print(words[:2])

    with open("/home/allan/nvim/learning/zig/wordle/src/sorted_words.txt", "w") as new_file:
        for word in words:
            new_file.write(word)

if __name__ == "__main__":
    sort_words("/home/allan/nvim/learning/zig/wordle/src/words.txt")
