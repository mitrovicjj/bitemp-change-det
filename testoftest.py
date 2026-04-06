from datasets import load_dataset
import pyarrow as pa
import pyarrow.ipc as ipc

HF_NAME = "blanchon/OSCD_MSI"
SPLIT = "train"

ds = load_dataset(HF_NAME)[SPLIT]
arrow_path = ds.cache_files[0]["filename"]

print("ARROW PATH:", arrow_path)

with pa.memory_map(arrow_path, "r") as source:
    reader = ipc.open_stream(source)
    table = reader.read_all()

print("\n=== TABLE ===")
print(table)

print("\n=== SCHEMA ===")
print(table.schema)

for col_name in ["image1", "image2", "mask"]:
    print(f"\n=== COLUMN: {col_name} ===")
    col = table.column(col_name)
    print("type:", col.type)
    print("num_chunks:", col.num_chunks)

    chunk0 = col.chunk(0)
    print("chunk0 type:", chunk0.type)
    print("chunk0 len :", len(chunk0))

    try:
        first = chunk0[0]
        print("first scalar type:", type(first))
        print("first scalar repr:", str(first)[:1000])
    except Exception as e:
        print("reading first scalar failed:", repr(e))