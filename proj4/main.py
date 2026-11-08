import sys
from partA import run_partA2, run_partA3, run_partA4
from partB import run_partB1, run_partB2, run_partB3, run_partB4

if __name__ == "__main__":
    path_to_imgs = "images"
    path_to_media = "media"
    # python main.py runs both parts, python main.py A or python main.py B just one
    part = sys.argv[1] if len(sys.argv) > 1 else "AB"
    if "A" in part:
        run_partA2(path_to_imgs, path_to_media)
        run_partA3(path_to_imgs, path_to_media)
        run_partA4(path_to_imgs, path_to_media)
    if "B" in part:
        run_partB1(path_to_imgs, path_to_media)
        run_partB2(path_to_imgs, path_to_media)
        run_partB3(path_to_imgs, path_to_media)
        run_partB4(path_to_imgs, path_to_media)
