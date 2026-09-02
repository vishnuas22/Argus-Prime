#!/usr/bin/env python
"""FaceForensics public data release downloader (user-approved copy, automated).

Non-interactive: TOS confirmed by possession of the emailed script + this run.
Example:
  .venv/bin/python deepfake/scripts/ffpp/ffpp_download.py /path/out -d compressed --sample_only
"""
import argparse
import os
import tempfile
from os.path import join

SERVER_URL = 'http://kaldir.vc.in.tum.de/FaceForensics/'
TOS_URL = SERVER_URL + 'webpage/FaceForensics_TOS.pdf'
BASE_URL = SERVER_URL + 'v1_cargo/'
ORIGINAL_VIDEOS_URL = BASE_URL + 'original_videos.tar.gz'
RELEASE_DATASET_SIZE = {'raw': '~3.5TB', 'compressed': '~130GB', 'images': '~135GB'}
DATASET_TYPES = ["raw", "compressed", "selfreenactment_raw",
                 "selfreenactment_compressed", "original_videos",
                 "selfreenactment_images", "source_to_target_images"]
NUM_SAMPLES = 5


def get_filelist(filelist_url):
    lines = __import__('urllib.request', fromlist=['urlopen']).urlopen(filelist_url)
    out = [l.decode('utf-8').rstrip('\n') for l in lines]
    return out


def download_file(url, out_file):
    out_dir = os.path.dirname(out_file)
    if not os.path.isfile(out_file):
        os.makedirs(out_dir, exist_ok=True)
        fh, tmp = tempfile.mkstemp(dir=out_dir)
        os.close(fh)
        __import__('urllib.request', fromlist=['urlretrieve']).urlretrieve(url, tmp)
        os.rename(tmp, out_file)
    else:
        print('WARNING: skipping existing ' + out_file)


def download_files(filenames, base_url, output_path, sample_only=False):
    os.makedirs(output_path, exist_ok=True)
    n = len(filenames) if not sample_only else NUM_SAMPLES
    for i, fn in enumerate(filenames):
        if i % 10 == 0:
            print(f"{i}/{n}", flush=True)
        download_file(base_url + fn, join(output_path, fn))
        if sample_only and i != 0 and i % (NUM_SAMPLES - 1) == 0:
            break
    print(f"{n}/{n}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('output_path')
    p.add_argument('-d', '--dataset_type', default='compressed')
    p.add_argument('--not_altered', action='store_true')
    p.add_argument('--not_original', action='store_true')
    p.add_argument('--not_mask', action='store_true')
    p.add_argument('--not_test', action='store_true')
    p.add_argument('--not_train', action='store_true')
    p.add_argument('--not_val', action='store_true')
    p.add_argument('--sample_only', action='store_true')
    args = p.parse_args()

    assert args.dataset_type in DATASET_TYPES, "bad dataset type"
    types = []
    if not args.not_altered: types.append('altered')
    if not args.not_original: types.append('original')
    if 'images' not in args.dataset_type and not args.not_mask: types.append('mask')

    folders = []
    if not args.not_test: folders.append('test')
    if not args.not_train: folders.append('train')
    if not args.not_val: folders.append('val')

    if 'selfreenactment' in args.dataset_type:
        dataset = 'selfreenactment'
        dtype = args.dataset_type.replace('selfreenactment_', '')
    else:
        dataset = 'source_to_target'
        dtype = args.dataset_type.replace('source_to_target_', '')

    print(f"dataset={args.dataset_type} size~{RELEASE_DATASET_SIZE.get(dtype, '?')} "
          f"folders={folders} types={types}", flush=True)

    if args.dataset_type == 'original_videos':
        print('~2gb single archive')
        download_file(ORIGINAL_VIDEOS_URL,
                      join(args.output_path, 'faceforensics_original_videos.tar.gz'))
        return 0

    for folder in folders:
        fl_folder = ('images_' + folder) if 'images' in args.dataset_type else folder
        filelist_url = BASE_URL + f'{dataset}/filelists/{fl_folder}.txt'
        filenames = get_filelist(filelist_url)
        print(f"\n== {folder} ({len(filenames)} files listed)", flush=True)
        for vt in types:
            outp = join(args.output_path, f'FaceForensics_{args.dataset_type}', folder, vt)
            base = BASE_URL + f'{dataset}/{dtype}/{folder}/{vt}/'
            print(f'{folder}/{vt} -> {outp}', flush=True)
            download_files(filenames, base, output_path=outp, sample_only=args.sample_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
