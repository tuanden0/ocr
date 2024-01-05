import os, sys
import shutil

'''
Docs:
https://sathishvj.medium.com/training-tesseract-ocr-for-a-new-font-and-input-set-on-mac-7622478cd3a1
https://saiashish90.medium.com/training-tesseract-ocr-with-custom-data-d3f4881575c0
https://www.youtube.com/watch?v=1v8BPw0Dn0I

Need:
Create folders: source, data, output, trained
'''

source_dir = r'./source'
data_dir = r'./data'
lang = "train"
font = "ocr"
part = f"{lang}.{font}.exp"
dest_dir = r'./trained'
output = r'./output'
logs = r'./log.txt'
unicharset_file = rf'{dest_dir}/unicharset'
font_properties = rf'{dest_dir}/font_properties'

def clearOldData(data_dir=data_dir):
    for f in os.listdir(data_dir):
        os.remove(os.path.join(data_dir, f))

def copyData(source_dir=source_dir):
    for f in os.listdir(source_dir):
        shutil.copyfile(os.path.join(source_dir, f), os.path.join(data_dir, f))

def labelingData(data_dir=data_dir):
    images = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.tif', '.bmp'))]
    print(f"{len(images)} number of images found")

    if len(images) == 0:
        sys.exit(0)

    for i, image in enumerate(images):
        ext = image.rsplit(".", 1)[1]
        filename = f"{part}{i}.{ext}"
        full_img_file = os.path.join(data_dir, filename)
        print(full_img_file)

        # Labeling img file name
        os.rename(os.path.join(data_dir, image), full_img_file)

        # Generate box data
        full_box_file = os.path.join(data_dir, f"{lang}.{font}.exp{i}")
        os.system(f"tesseract {full_img_file} {full_box_file} batch.nochop makebox")

def trainData(data_dir=data_dir):
    files = os.listdir(data_dir)
    for item in files:
        if not item.endswith(('.jpeg', '.box')):
            os.remove(os.path.join(data_dir, item))

    # Generating the tuples of filenames
    files = os.listdir(data_dir)
    jpgs = sorted([x for x in files if x.endswith('.jpeg')])
    boxes = sorted([x for x in files if x.endswith('.box')])
    trainfiles = list(zip(jpgs, boxes))

    # generating TR files and unicode charecter extraction
    unicharset = f"unicharset_extractor --output_unicharset {unicharset_file} "
    unicharset_args = f""

    errorfiles = []
    for image, box in trainfiles:
        unicharset_args += f"{os.path.join(data_dir, box)} "

        img_name = image.rsplit(".", 1)[0]

        if os.path.isfile(f"{dest_dir}/{img_name}.tr"):
            continue
        try:
            os.system(f"tesseract {data_dir}/{image} {dest_dir}/{img_name} nobatch box.train")
        except:
            errorfiles.append((image, box))

    os.system(unicharset+unicharset_args)

    # Creating font proerties file
    with open(font_properties, 'w') as f:
        f.write(f"{font} 0 0 0 1 0")

    # # Getting all .tr files and training
    trfiles = [os.path.join(dest_dir, f) for f in os.listdir(dest_dir) if f.endswith('.tr')]
    mftraining = f"mftraining -F {font_properties} -U {unicharset_file} -O {output}/{lang}.unicharset -D {output}"
    cntraining = f"cntraining -D {output}"
    for file in trfiles:
        mftraining += f" {file}"
        cntraining += f" {file}"

    os.system(mftraining)
    os.system(cntraining)

    # # # Renaming training files and merging them
    os.chdir(output)
    os.rename('inttemp', f'{lang}.inttemp')
    os.rename('normproto', f'{lang}.normproto')
    os.rename('pffmtable', f'{lang}.pffmtable')
    os.rename('shapetable', f'{lang}.shapetable')
    os.system(f"combine_tessdata {lang}.")

    os.chdir("..")

    # Writing log file
    if len(errorfiles) == 0:
        return

    with open(logs, 'w+') as f:
        f.write('\n'.join('%s %s' % x for x in errorfiles))

if __name__ == '__main__':
    clearOldData(data_dir)
    copyData(source_dir)
    labelingData(data_dir)
    trainData(data_dir)
