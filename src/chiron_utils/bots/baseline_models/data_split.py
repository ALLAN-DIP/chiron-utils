import os

class Splitter:
    def __init__(self, src_path, dest_path, ratio_split=True, split_points=[0.9, 1], split_names=["train.jsonl", "test.jsonl"], total_games=1000):
        self.src_path = src_path
        self.dest_path = dest_path
        self.split_names = split_names
        
        if ratio_split:
            self.split_points = list(point * total_games for point in split_points)
            self.total_games = total_games
        else:
            self.split_points = split_points
            self.total_games = split_points[-1]
        
    
    def split(self):
        write_files = list()
        for name in self.split_names:
            filename = os.path.join(self.dest_path, name)
            write_files.append(open(filename, 'w'))

        with open(self.src_path, 'r') as src:
            for n, line in enumerate(src):
                for s, split in enumerate(self.split_points):
                    if n < split:
                        write_files[s].write(line)
                        print(f"Writing line {n} in set {self.split_names[s]}")
                        break
                if n >= self.total_games:
                    break

        for file in write_files:
            file.close()

if __name__ == "__main__":
    src_path = os.path.join("D:", os.sep, "Downloads", "dipnet-data-diplomacy-v1-27k-msgs", "standard_no_press.jsonl")
    dest_path = os.path.join("D:", os.sep, "Downloads", "dipnet-data-diplomacy-v1-27k-msgs", "test")
    splitter = Splitter(src_path, dest_path, ratio_split=False, split_points=[100, 200], split_names=["test.jsonl", "train.jsonl"], total_games=1100)
    splitter.split()