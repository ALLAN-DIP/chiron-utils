import os
import json

def main():
    path = os.path.join("D:", os.sep, "Downloads", "dipnet-data-diplomacy-v1-27k-msgs")
    filepath = os.path.join(path, "standard_no_press.jsonl")
    messages_count = 0
    
    with open(filepath, 'r') as file:
        codes = set()
        for i, line in enumerate(file):
            entry = json.loads(line)
            # print(entry.keys())
            # print(entry["map"])
            # print(entry["rules"])
            for j, phase in enumerate(entry["phases"]):
                # print(phase['state'].keys())
                print(phase['state']['name'])
                # print(phase['state']['units'])
                if j == 16:
                    break
                # print(phase["orders"])
                # print(phase["orders"].keys())
            """
                influences = phase["state"]["centers"]
                for infs in influences.values():
                    for inf in infs:
                        codes.add(inf)
            """
            if i == 1:
                break


        print(messages_count)
        

if __name__ == "__main__":
    main()