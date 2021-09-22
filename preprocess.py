import glob
import json

files = glob.glob("./data/*.txt")


all_texts = []


for doc_id, file in enumerate(files):
    print("Printing for file ----------------------------------------", file)
    with open(file) as f:
        raw_text = f.read()
        texts = raw_text.split("ADVERTISEMENT\n\nContinue reading the main story")

        date_arr = file[-9:][:-4].split("_")
        date_arr.append("2020")
        date_string = "/".join(date_arr)

        for text_id, _text in enumerate(texts):
            # print("Text ####################", text_id)
            # print(_text)

            all_texts.append({
                "doc_id": doc_id,
                "text_id": text_id,
                "filename": file,
                "text": _text,
                "date": date_string
            }
            )

print(all_texts)

with open("./data/data.json", "w") as f:
    f.write(json.dumps(all_texts, indent=4))

