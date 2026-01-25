from pathlib import Path
from box import ConfigBox
from datasets import load_dataset


from src.utils.folder_management import create_folders
from src.utils.decorator import parser


@parser(prog_name="Transform Pipeline", dscr="Split Naturally the data | Patchify it")
def main(params: ConfigBox) -> None:
    data = load_dataset(params.hf_name, split='train')
    data.save_to_disk(params.data_dir)
    

if __name__ == "__main__":
    main()
    
