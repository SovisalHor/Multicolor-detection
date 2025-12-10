import os
import glob

labels = glob.glob(r"D:\MultiColor.v2-mcolor.yolov8\**\labels\*.txt", recursive=True)

bad = []

for file in labels:
    with open(file, "r") as f:
        for line in f.readlines():
            if not line.strip():
                continue
            cls = int(line.split()[0])
            if cls > 2:   # <-- you only have 0,1,2
                bad.append((file, cls))

if bad:
    print("❌ Found invalid class IDs:")
    for b in bad:
        print(b)
else:
    print("✅ All labels OK")
