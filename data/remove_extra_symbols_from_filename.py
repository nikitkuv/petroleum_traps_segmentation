import os


folder = "data/cps/"
suffixes = [".cps3", "-UNIQ1"]

for filename in os.listdir(folder):
    old_path = os.path.join(folder, filename)

    if not os.path.isfile(old_path):
        continue

    new_name = filename

    for suffix in suffixes:
        if new_name.endswith(suffix):
            new_name = new_name[:-len(suffix)]

    if new_name != filename:
        new_path = os.path.join(folder, new_name)
        print(f"Renaming: {filename} -> {new_name}")
        os.rename(old_path, new_path)

print("Done")
