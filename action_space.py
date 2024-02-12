import ocrspace
from os import listdir
from os.path import isfile, join

# Define config
in_dir = r".\dataset\processed"
out_dir = r".\results\space"
api_key = r"key"
lang = ocrspace.Language.Vietnamese
api = ocrspace.API(
    api_key=api_key,
    language=lang,
    OCREngine=3,
)

def get_files():
    files = []
    for f in listdir(in_dir):    
        img = join(in_dir, f)
        if isfile(img):
            p = f.rsplit(".", 1)[0]
            txt = join(out_dir, f"{p}.txt")
            files.append({
                "img": img, 
                "txt": txt,
                "name": p
            })

    return files

def ocr(img):
    print(img)
    return api.ocr_file(img)

def main():
    files = get_files()
    if len(files) == 0:
       return
    
    for file in files:
        img = file.get("img", "")
        if img == "":
            continue

        text = ocr(img)
        n_text = len(text)
        if n_text == 0:
           continue
        
        out = file.get("txt", "")
        if out == "":
           continue

        print(f"{img}: {text}")
        with open(out, "w", encoding="utf-8") as f:
            if isinstance(text, str):
                f.write(text)
            
            if type(text) is list:
                lines = ""
                i = 0
                for t in text:
                    lines += t
                    if i != n_text-1:
                        lines += "\\N"
                    i += 1
                
                f.write(lines)

if __name__ == "__main__":
    try:
        main()
        # print(api.ocr_file(r'./dataset/processed/0_00_09_551__0_00_10_801_1000006311280007212800720.jpeg'))
    except Exception as e:
        print(e)
    except KeyboardInterrupt:
        pass
