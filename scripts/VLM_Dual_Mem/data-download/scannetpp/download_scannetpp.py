"""
Download ScanNet++ data

Default: download splits with scene IDs and default files
that can be used for novel view synthesis on DSLR and iPhone images
and semantic tasks on the mesh
"""

import argparse
import time
from pathlib import Path
import urllib.request
from urllib.request import urlretrieve
import urllib.error
import yaml
from munch import Munch
from tqdm import tqdm
import json
import os
import zipfile

from scene_release import ScannetppScene_Release


def read_txt_list(path):
    with open(path) as f:
        lines = f.read().splitlines()

    return lines


def load_json(path):
    with open(path) as f:
        j = json.load(f)

    return j


def load_yaml_munch(path):
    with open(path) as f:
        y = yaml.load(f, Loader=yaml.Loader)

    return Munch.fromDict(y)


def check_remote_file_exists(url):
    """
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    """
    request = urllib.request.Request(url)
    request.get_method = lambda: "HEAD"

    try:
        urllib.request.urlopen(request)
        return True
    except urllib.request.HTTPError:
        return False


def download_scannetpp_gs(cfg, scene_ids):
    '''
    Download ScanNet++GS data
    '''

    print('Downloading ScanNet++GS data...')
    for scene_id in tqdm(scene_ids, desc="scenes"):
        src_path = Path('scannetpp_gs') / scene_id / 'ckpts' / 'point_cloud_30000.ply'
        tgt_path = Path(cfg.scannetpp_gs_dir) / f"{scene_id}/point_cloud_30000.ply"
        if not check_download_file(cfg, cfg.scannetpp_gs_url, src_path, tgt_path, cfg.dry_run):
            break


def urlretrieve_multi_trials(url, filename, max_trials=5):
    # 401 -> invalid token or token expired
    # 404 -> File not found
    # 406 -> Not acceptable, please update the download script
    for i in range(max_trials):
        try:
            urlretrieve(url, filename)
            time.sleep(0.2)     # wait a bit to prevent being rejected for frequent access to the server
            return True

        except urllib.error.ContentTooShortError as e:
            # Failed to download the file completely. Delete the file and retry.
            # Delete filename
            print("ERROR: Content too short. It is likely that the download was incomplete due to network issues. Retrying...")
            if i < max_trials - 1:
                if Path(filename).exists():
                    os.remove(filename)
                time.sleep(0.5)  # wait a bit before retrying
            else:
                print(f"Failed to download {url} after {max_trials} trials")
                raise e

        # HTTP error
        except urllib.error.HTTPError as e:
            if e.code == 401:
                print(f"ERROR when accessing {url}")
                print("It could be that an invalid or expired token is used.")
                raise e

            elif e.code == 404:
                print(f"ERROR when accessing {url}")
                raise e

            elif e.code == 406:
                message = str(e.read())
                print(f"ERROR when accessing {url}")
                print(e.code, message)
                print("Please update the download script and try again")
                raise e
            else:
                print(f"ERROR when accessing {url}")
                print(e.code, e.read())
                raise e
    return False


def download_file(url, filename, verbose=True, make_parent=False):
    """
    Download file from url to filename
    """
    # download_url = url.

    if make_parent:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"{url} ==> {filename}")
    return urlretrieve_multi_trials(url, filename)


def check_download_file(cfg, url_template, remote_path, local_path, dry_run):
    """
    check if file exists, else download
    remote_path: relative to root
    local_path: full path
    dry_run: only check if file exists, don't download
    """
    # Updates: fix Windows users that use backslash for remote_path
    remote_path = str(remote_path).replace("\\", "/")
    url = url_template.replace("TOKEN", cfg.token).replace("FILEPATH", remote_path)

    if dry_run:
        status = check_remote_file_exists(url)
        if status:
            print("Remote file exists:", url)
        else:
            print("Remote file missing:", url)
        return status

    if local_path.is_file():
        if cfg.verbose:
            print("File exists, skipping download: ", local_path)
        return True
    else:
        return download_file(url, local_path, verbose=cfg.verbose, make_parent=True)


def main(args):
    cfg = load_yaml_munch(args.config_file)
    # Ask the user to provide the token if not provided
    if cfg.get("token", "<YOUR_TOKEN_HERE>") == "<YOUR_TOKEN_HERE>":
        cfg.token = input("Please enter your download token: ").strip()
        if cfg.token == "":
            print("No token provided, exiting. Please apply for a token from ScanNet++ official website.")
            return
    else:
        print(f"Using token from config file: {cfg.token}")

    if cfg.get("data_root", "<DOWNLOAD_LOCATION_HERE>") == "<DOWNLOAD_LOCATION_HERE>":
        cfg.data_root = input("Please enter your download location: (default: ./scannetpp_data)").strip()
        if cfg.data_root == "":
            cfg.data_root = "./scannetpp_data"
    print(f"Downloading to: {cfg.data_root}")

    # Notify user the size of the whole dataset
    print(
        "WARNING: Downloading the full ScanNet++ dataset with default splits and assets "
        "will require approximately 1.5TB of disk space.\n"
        "If you have already customized splits or assets in your config file, the required space may differ.\n"
        "Do you want to proceed with the download? (y/n)\n > ", end=""
    )
    ans = input().strip().lower()
    if ans != 'y':
        print("Exiting.")
        return

    if cfg.dry_run:
        print("Dry run: check if remote files exist, no files will be downloaded")

    missing = []

    data_root = Path(cfg.data_root)

    # create data root directory
    data_root.mkdir(parents=True, exist_ok=True)

    # download meta files
    for path in cfg.meta_files:
        if not check_download_file(cfg, cfg.root_url, path, data_root / path, cfg.dry_run):
            missing.append(str(data_root / path))

    if cfg.metadata_only:
        print("Downloaded metadata, done.")
        return

    # read all the split files
    split_lists = {}
    for split in cfg.splits:
        split_path = data_root / "splits" / f"{split}.txt"
        split_lists[split] = read_txt_list(split_path)

    # get the list of scenes to be downloaded
    if cfg.get("download_scenes"):
        scene_ids = cfg.download_scenes
    elif cfg.get("download_splits"):
        scene_ids = []
        for split in cfg.download_splits:
            split_path = Path(cfg.data_root) / "splits" / f"{split}.txt"
            scene_ids += read_txt_list(split_path)

    # we know the scene ids, check for 3rd party datasets
    if cfg.get("scannetpp_gs_dir"):
        download_scannetpp_gs(cfg, scene_ids)
        print(f"Downloaded ScanNet++GS data to {cfg.scannetpp_gs_dir}, done.")
        return

    # get the list of assets to download for these scenes
    if cfg.get("download_assets"):
        download_assets = cfg.download_assets
    elif cfg.get("download_options"):
        download_assets = []
        for option in cfg.download_options:
            option_assets = cfg.option_assets[option]
            for asset in option_assets:
                if asset not in download_assets:
                    download_assets.append(asset)
    else:
        download_assets = cfg.default_assets

    print("Downloading assets:", download_assets)
    print("Scenes selected: ", len(scene_ids))

    failed_scenes = []
    skipped_count = 0
    downloaded_count = 0
    
    for scene_id in tqdm(scene_ids, desc="scenes"):
        scene_has_error = False
        # download from here
        # path relative to root url
        src_scene = ScannetppScene_Release(scene_id, data_root="data")
        # to here
        tgt_scene = ScannetppScene_Release(scene_id, data_root=Path(cfg.data_root) / "data")

        # get the split for this scene
        split = None
        for split in cfg.splits:
            if scene_id in split_lists[split]:
                break
        assert split is not None, f"Scene {scene_id} not in any split"

        for asset in tqdm(download_assets, desc="assets", leave=False):
            # some assets not present in test splits
            if asset in cfg.exclude_assets.get(split, []):
                continue

            # check if asset is zipped, download the zip and unzip it to the target path
            if asset in cfg.zipped_assets:
                tgt_path = getattr(tgt_scene, asset)

                # Check if already extracted (directory or file exists)
                if tgt_path.is_file() or tgt_path.is_dir():
                    if cfg.verbose:
                        print(f"File/directory exists, skipping download: {tgt_path}")
                    skipped_count += 1
                    continue

                src_download_path = getattr(src_scene, asset).with_suffix(".zip")
                tgt_download_path = tgt_path.with_suffix(".zip")

                # Check if zip file exists and is valid
                if tgt_download_path.is_file():
                    try:
                        # Try to open the zip to verify it's complete
                        with zipfile.ZipFile(tgt_download_path, "r") as test_zip:
                            test_zip.testzip()  # Returns None if valid, or first bad file if invalid
                        # If we get here, zip is valid - extract it
                        if not cfg.dry_run:
                            if cfg.verbose:
                                print(f"Found existing zip file, extracting: {tgt_download_path}")
                            try:
                                with zipfile.ZipFile(tgt_download_path, "r") as zip_ref:
                                    zip_ref.extractall(tgt_download_path.parent)
                                # remove the zip file
                                if cfg.verbose:
                                    print(f"Delete zip file: {tgt_download_path}")
                                tgt_download_path.unlink()
                                downloaded_count += 1
                            except Exception as e:
                                print(f"Error extracting {tgt_download_path}: {e}")
                                # Delete corrupted zip and re-download
                                tgt_download_path.unlink()
                                if not check_download_file(cfg, cfg.root_url, src_download_path, tgt_download_path, cfg.dry_run):
                                    missing.append(str(tgt_download_path))
                                    scene_has_error = True
                                    continue
                                # Try to extract again after re-download
                                if not cfg.dry_run:
                                    try:
                                        with zipfile.ZipFile(tgt_download_path, "r") as zip_ref:
                                            zip_ref.extractall(tgt_download_path.parent)
                                        tgt_download_path.unlink()
                                        downloaded_count += 1
                                    except Exception as e2:
                                        print(f"Error extracting after re-download {tgt_download_path}: {e2}")
                                        missing.append(str(tgt_download_path))
                                        scene_has_error = True
                        else:
                            skipped_count += 1
                        continue
                    except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
                        # Zip file is corrupted, delete and re-download
                        print(f"Corrupted zip file detected, will re-download: {tgt_download_path}")
                        tgt_download_path.unlink()

                # Download the zip file
                if not check_download_file(cfg, cfg.root_url, src_download_path, tgt_download_path, cfg.dry_run):
                    missing.append(str(tgt_download_path))
                    scene_has_error = True
                    continue  # Continue with next asset instead of breaking

                if not cfg.dry_run:
                    # unzip it
                    try:
                        if cfg.verbose:
                            print(f"Unzipping: {tgt_download_path}")
                        with zipfile.ZipFile(tgt_download_path, "r") as zip_ref:
                            zip_ref.extractall(tgt_download_path.parent)
                        # remove the zip file
                        if cfg.verbose:
                            print(f"Delete zip file: {tgt_download_path}")
                        tgt_download_path.unlink()
                        downloaded_count += 1
                    except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
                        print(f"Error: Corrupted zip file {tgt_download_path}: {e}")
                        print(f"Will retry download on next run")
                        missing.append(str(tgt_download_path))
                        scene_has_error = True
                        # Don't delete corrupted zip, let it be re-downloaded next time
            else:
                #  download single file
                src_path = getattr(src_scene, asset)
                tgt_path = getattr(tgt_scene, asset)
                
                # Check if file already exists
                if tgt_path.is_file():
                    if cfg.verbose:
                        print(f"File exists, skipping download: {tgt_path}")
                    skipped_count += 1
                    continue
                
                if not check_download_file(cfg, cfg.root_url, src_path, tgt_path, cfg.dry_run):
                    missing.append(str(tgt_path))
                    scene_has_error = True
                    continue  # Continue with next asset instead of breaking
                else:
                    downloaded_count += 1

        if scene_has_error:
            failed_scenes.append(scene_id)
            print(f"\nWarning: Some assets failed for scene {scene_id}, continuing with next scene...")
        else:
            if cfg.verbose:
                print(f"Successfully completed scene {scene_id}")

    # Print summary
    print("\n" + "="*60)
    print("Download Summary:")
    print(f"  Total scenes processed: {len(scene_ids)}")
    print(f"  Successfully downloaded: {downloaded_count} assets")
    print(f"  Skipped (already exists): {skipped_count} assets")
    print(f"  Failed scenes: {len(failed_scenes)}")
    if failed_scenes:
        print(f"  Failed scene IDs: {failed_scenes[:10]}{'...' if len(failed_scenes) > 10 else ''}")
    print(f"  Missing files: {len(missing)}")
    print("="*60)
    
    if missing:
        print(f"\n{len(missing)} files failed to download or are incomplete.")
        print("You can re-run this script to retry failed downloads.")
        print("The script will automatically skip files that are already complete.")
        if len(missing) <= 20:
            print("Missing files:")
            for m in missing:
                print(f"  - {m}")
    else:
        print("\nDownload successful! All files downloaded and extracted.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_file", help="Path to config file")
    args = p.parse_args()

    main(args)
