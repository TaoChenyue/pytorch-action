import argparse
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo path:https://github.com/TaoChenyue/pytorch-action/tree/main/test"
    )
    # dataset
    parser.add_argument('--root',type=str,default="datasets/",help='dataset root path')
    parser.add_argument("--split_rate",type=float,default=0.7,help="rate for split dateset into train")
    parser.add_argument("--num_classes",type=int,default=10,help="how many classes of dataset")
    # train
    parser.add_argument("--gpus",type=str,default='',help="like 0,1,2 . empty means cpu")
    parser.add_argument("--epochs", type=int, default=100,help="train epochs")
    parser.add_argument("--batch_size",type=int,default=16,help="batch size")
    parser.add_argument("--lr",type=float,default=0.001,help="learn rate")
    parser.add_argument("--ckpt",type=str,help="path of checkpoint")
    # save
    parser.add_argument("--save_name",type=str,default="default",help="save model name")
    parser.add_argument("--save_suffix",type=str,default="pth",help="suffix of save model")
    parser.add_argument("--config", type=str,help="config yaml/yml file path")
    args= parser.parse_args()
    if args.config:
        with open(args.config) as f:
            content = yaml.safe_load(f)
        parser.set_defaults(**content)
    args = parser.parse_args()
    return args
    
def main():
    print(parse_args())

if __name__ == "__main__":
    main()

